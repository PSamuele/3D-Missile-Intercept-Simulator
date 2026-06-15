#!/usr/bin/env python3
"""
missile_intercept_sim.py
========================
Simulazione 3D di intercettazione missilistica per ingegneria aeronautica.

Fisica modellata
----------------
- Bersaglio : punto-massa in volo 3D con manovre casuali ma realistiche
               (virate coordinate, salita/discesa), vincolate da nz_max.
- Missile   : punto-massa con guida a Navigazione Proporzionale Vera (TPN),
               fase boost + coast, compensazione gravitazionale.
- Integrazione: Euler a passo fisso (dt=0.05 s, sufficiente per questa scala).

Avvio rapido
------------
    python missile_intercept_sim.py                   # parametri di default
    python missile_intercept_sim.py --seed 42         # run riproducibile
    python missile_intercept_sim.py --nz-max 7 --target-speed 300
    python missile_intercept_sim.py --help            # tutte le opzioni

Struttura del codice
--------------------
  TargetConfig / MissileConfig / SimConfig  ← dataclass di configurazione
  Target      ← dinamica bersaglio
  Missile     ← guida PN + propulsione
  InterceptionSimulation ← orchestratore
  Visualizer  ← grafici multi-panel
  main()      ← CLI (argparse)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registra proiezione 3D

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

G: float = 9.81  # accelerazione gravitazionale [m/s²]


# ══════════════════════════════════════════════════════════════════════════════
# Configurazioni
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TargetConfig:
    """
    Parametri fisici e prestazionali del bersaglio aereo.

    Tutti i valori sono modificabili a runtime (CLI o istanziazione diretta).
    """

    mass_kg: float = 15_000.0
    """Massa del velivolo [kg]. Influisce sul fattore di carico strutturale."""

    speed_mps: float = 250.0
    """Velocità di crociera [m/s] (≈ 900 km/h, ≈ Mach 0.75 a livello del mare)."""

    nz_max: float = 5.0
    """Fattore di carico massimo strutturale [-]. Limita la velocità di rollio."""

    altitude_min_m: float = 3_000.0
    """Quota minima di volo [m AGL]."""

    altitude_max_m: float = 12_000.0
    """Quota massima di volo [m AGL]."""

    maneuver_period_s: float = 20.0
    """Periodo medio tra cambi di manovra (processo di Poisson) [s].
    Valori più bassi → bersaglio più aggressivo."""

    heading_rate_max_rads: float = 0.08
    """Velocità massima di cambio rotta [rad/s] (≈ 4.6°/s)."""

    climb_rate_max_mps: float = 25.0
    """Rateo di salita/discesa massimo sostenuto [m/s]."""

    spawn_range_km: float = 40.0
    """Raggio massimo di comparsa del bersaglio dall'origine [km]."""

    spawn_range_min_km: float = 15.0
    """Raggio minimo di comparsa [km] (evita spawn troppo vicino)."""


@dataclass
class MissileConfig:
    """
    Parametri fisici e prestazionali del missile intercettore.
    """

    mass_kg: float = 300.0
    """Massa al lancio [kg]."""

    thrust_N: float = 60_000.0
    """Spinta motore nella fase boost [N]."""

    burn_time_s: float = 5.0
    """Durata della combustione [s]."""

    speed_max_mps: float = 900.0
    """Velocità massima [m/s] (≈ Mach 2.6 a livello del mare)."""

    accel_max_lat_mps2: float = 300.0
    """Accelerazione laterale massima [m/s²] (≈ 30g). Limite strutturale del missile."""

    nav_constant: float = 4.0
    """Rapporto di navigazione effettivo N' (Proportional Navigation).
    Valori tipici: 3–5. Maggiore → convergenza più rapida, ma più sensibile al rumore."""

    launch_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    """Posizione di lancio [m] — punto fisso a terra (x, y, z=0)."""

    launch_speed_mps: float = 80.0
    """Velocità iniziale al momento del lancio [m/s]."""


@dataclass
class SimConfig:
    """
    Parametri dell'integrazione numerica e terminazione della simulazione.
    """

    dt: float = 0.05
    """Passo temporale [s]. Riduci per maggiore accuratezza (più lento)."""

    t_max: float = 120.0
    """Tempo massimo di simulazione [s]."""

    kill_radius_m: float = 30.0
    """Raggio di kill [m]. Intercettazione dichiarata se distanza < valore."""

    seed: Optional[int] = None
    """Seed casuale per riproducibilità. None = casuale ogni run."""

    animate: bool = False
    """Abilita animazione 3D (più lenta ma visivamente interessante)."""

    save_fig: Optional[str] = None
    """Se specificato, salva la figura in questo path (es. 'sim.png')."""


# ══════════════════════════════════════════════════════════════════════════════
# Target
# ══════════════════════════════════════════════════════════════════════════════

class Target:
    """
    Bersaglio aereo con dinamica 3D realistica.

    Modello: punto-massa in volo coordinato.
    Manovre: cambi di rotta casuali + salita/discesa, con:
      - rate-limiting della velocità angolare di imbardata
      - vincolo di fattore di carico nz_max (virata coordinata)
      - quote vincolate a [altitude_min_m, altitude_max_m]

    La rotta iniziale è orientata approssimativamente verso l'origine ±45°
    per garantire un ingaggio realistico.
    """

    def __init__(self, cfg: TargetConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        # ── Posizione iniziale casuale ─────────────────────────────────
        r_km = rng.uniform(cfg.spawn_range_min_km, cfg.spawn_range_km)
        r_m = r_km * 1_000.0
        azimuth = rng.uniform(0.0, 2 * np.pi)

        x0 = r_m * np.cos(azimuth)
        y0 = r_m * np.sin(azimuth)
        z0 = rng.uniform(cfg.altitude_min_m, cfg.altitude_max_m)

        self.pos: np.ndarray = np.array([x0, y0, z0], dtype=float)

        # Rotta iniziale: punta verso l'origine ±45° (ingaggio più realistico)
        toward_origin = np.arctan2(-y0, -x0)
        heading0 = toward_origin + rng.uniform(-np.pi / 4, np.pi / 4)
        self.heading: float = heading0
        self.vz: float = 0.0

        self.vel: np.ndarray = self._rebuild_vel()

        # ── Stato manovre ──────────────────────────────────────────────
        self._next_maneuver_t: float = 0.0
        self._tgt_heading: float = heading0
        self._tgt_vz: float = 0.0

        # ── Storie ────────────────────────────────────────────────────
        self.pos_history: list[np.ndarray] = [self.pos.copy()]
        self.vel_history: list[np.ndarray] = [self.vel.copy()]
        self.nz_history: list[float] = [1.0]

        logger.info(
            f"[Target] Comparso a ({x0/1000:.1f}, {y0/1000:.1f}, {z0:.0f}m), "
            f"range={r_km:.1f}km, rotta iniziale={np.degrees(heading0):.0f}°"
        )

    # ── Metodi privati ─────────────────────────────────────────────────────

    def _rebuild_vel(self) -> np.ndarray:
        """Ricostruisce il vettore velocità da heading, vz e speed."""
        v_h = np.sqrt(max(
            self.cfg.speed_mps ** 2 - self.vz ** 2,
            (0.4 * self.cfg.speed_mps) ** 2,
        ))
        return np.array([
            v_h * np.cos(self.heading),
            v_h * np.sin(self.heading),
            self.vz,
        ])

    def _schedule_maneuver(self, t: float) -> None:
        """Pianifica la prossima manovra casuale."""
        wait = self.rng.exponential(self.cfg.maneuver_period_s)
        self._next_maneuver_t = t + wait

        delta_hdg = self.rng.uniform(-np.pi / 2, np.pi / 2)
        self._tgt_heading = self.heading + delta_hdg

        self._tgt_vz = self.rng.uniform(
            -self.cfg.climb_rate_max_mps,
            self.cfg.climb_rate_max_mps,
        )
        logger.debug(
            f"[Target] t={t:.1f}s — Nuova manovra: "
            f"Δhdg={np.degrees(delta_hdg):.1f}°, vz={self._tgt_vz:.1f}m/s "
            f"→ prossima fra {wait:.1f}s"
        )

    # ── Interfaccia pubblica ───────────────────────────────────────────────

    def step(self, t: float, dt: float) -> np.ndarray:
        """Avanza lo stato del bersaglio di dt secondi. Ritorna la nuova posizione."""

        # Pianifica manovra se necessario
        if t >= self._next_maneuver_t:
            self._schedule_maneuver(t)

        speed_h = np.hypot(self.vel[0], self.vel[1])
        speed_h = max(speed_h, self.cfg.speed_mps * 0.5)

        # ── Limite omega da fattore di carico (virata coordinata) ──────
        # In virata coordinata: L = nz * W → nz = V²/(g·R)
        # ω_max dal fattore di carico: ω = (nz-1)*g / V
        omega_nz = (self.cfg.nz_max - 1.0) * G / speed_h
        omega_max = min(self.cfg.heading_rate_max_rads, omega_nz)

        # Cambio rotta limitato al rateo max
        delta_hdg_raw = (self._tgt_heading - self.heading + np.pi) % (2 * np.pi) - np.pi
        d_hdg = float(np.clip(delta_hdg_raw, -omega_max * dt, omega_max * dt))
        self.heading += d_hdg

        # Fattore di carico effettivo (per telemetria)
        omega_actual = abs(d_hdg) / dt if dt > 0 else 0.0
        nz_actual = 1.0 + speed_h * omega_actual / G
        self.nz_history.append(float(np.clip(nz_actual, 1.0, self.cfg.nz_max + 0.5)))

        # ── Velocità verticale ─────────────────────────────────────────
        vz_accel = 10.0  # [m/s²] — rateo di variazione del rateo di salita
        self.vz += float(np.clip(self._tgt_vz - self.vz, -vz_accel * dt, vz_accel * dt))

        # ── Quota vincolata ────────────────────────────────────────────
        next_z = self.pos[2] + self.vz * dt
        if next_z < self.cfg.altitude_min_m:
            self.vz = 0.0
            self._tgt_vz = abs(self._tgt_vz) * 0.5
            next_z = self.cfg.altitude_min_m
        elif next_z > self.cfg.altitude_max_m:
            self.vz = 0.0
            self._tgt_vz = -abs(self._tgt_vz) * 0.5
            next_z = self.cfg.altitude_max_m

        # ── Ricostruzione velocità e integrazione posizione ────────────
        self.vel = self._rebuild_vel()
        self.pos = np.array([
            self.pos[0] + self.vel[0] * dt,
            self.pos[1] + self.vel[1] * dt,
            next_z,
        ])

        self.pos_history.append(self.pos.copy())
        self.vel_history.append(self.vel.copy())
        return self.pos

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


# ══════════════════════════════════════════════════════════════════════════════
# Missile
# ══════════════════════════════════════════════════════════════════════════════

class Missile:
    """
    Missile intercettore a terra con guida a Navigazione Proporzionale Vera (TPN).

    Legge di guida: True Proportional Navigation
    ─────────────────────────────────────────────
    Dati:
        r⃗  = pos_bersaglio - pos_missile     (vettore LOS)
        v⃗_rel = vel_bersaglio - vel_missile  (velocità relativa)

    Velocità di chiusura:
        Vc = -d|r|/dt ≈ -dot(r̂, v⃗_rel)     (positiva se in avvicinamento)

    Velocità angolare del LOS (vettore):
        ω⃗_LOS = (r⃗ × v⃗_rel) / |r|²

    Accelerazione comandata (perpendicolare al LOS):
        a⃗_cmd = N' · Vc · (ω⃗_LOS × r̂_LOS)

    A questa si aggiunge la compensazione gravitazionale [0, 0, +g]
    per mantenere la guida accurata nella fase di coast.

    Fasi operative
    ──────────────
    BOOST  (0 … burn_time_s): spinta motore lungo il vettore velocità
    COAST  (dopo burn_time):  guida PN pura + gravity comp
    """

    def __init__(self, cfg: MissileConfig) -> None:
        self.cfg = cfg
        self.pos: np.ndarray = cfg.launch_pos.copy().astype(float)
        self.vel: np.ndarray = np.zeros(3, dtype=float)

        self.active: bool = False
        self._t_launch: float = 0.0

        # Storie
        self.pos_history: list[np.ndarray] = [self.pos.copy()]
        self.vel_history: list[np.ndarray] = [self.vel.copy()]
        self.accel_lat_history: list[float] = [0.0]   # [m/s²]
        self.closing_vel_history: list[float] = [0.0]  # [m/s]

    def launch(self, initial_target_pos: np.ndarray, t0: float = 0.0) -> None:
        """Lancia il missile verso la posizione iniziale del bersaglio."""
        r_vec = initial_target_pos - self.pos
        r_hat = r_vec / np.linalg.norm(r_vec)
        self.vel = r_hat * self.cfg.launch_speed_mps
        self.active = True
        self._t_launch = t0

        logger.info(
            f"[Missile] Lancio! Range iniziale: "
            f"{np.linalg.norm(r_vec)/1000:.1f} km | "
            f"Quota bersaglio: {initial_target_pos[2]:.0f} m"
        )

    def step(
        self,
        t: float,
        dt: float,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> np.ndarray:
        """
        Avanza lo stato del missile di dt secondi.

        Args:
            t          : tempo corrente [s]
            dt         : passo temporale [s]
            target_pos : posizione corrente del bersaglio [m]
            target_vel : velocità corrente del bersaglio [m/s]

        Returns:
            Nuova posizione del missile [m]
        """
        if not self.active:
            return self.pos

        # ── Cinematica relativa ────────────────────────────────────────
        r_vec = target_pos - self.pos   # vettore da missile a bersaglio
        r = float(np.linalg.norm(r_vec))

        if r < 0.5:
            self.active = False
            return self.pos

        r_hat = r_vec / r
        v_rel = target_vel - self.vel   # velocità relativa (bersaglio - missile)

        # Velocità di chiusura (positiva = avvicinamento)
        Vc = float(-np.dot(r_hat, v_rel))

        # Velocità angolare del LOS [rad/s] — vettore
        omega_los: np.ndarray = np.cross(r_vec, v_rel) / (r ** 2)

        # ── Guida PN ──────────────────────────────────────────────────
        # a_cmd = N' * Vc * (ω_LOS × r̂)
        a_guidance: np.ndarray = (
            self.cfg.nav_constant * Vc * np.cross(omega_los, r_hat)
        )

        # Compensazione gravitazionale (mantiene precisione in coast)
        a_grav_comp: np.ndarray = np.array([0.0, 0.0, G])

        a_lat_cmd: np.ndarray = a_guidance + a_grav_comp

        # Saturazione accelerazione laterale
        a_lat_mag = float(np.linalg.norm(a_lat_cmd))
        if a_lat_mag > self.cfg.accel_max_lat_mps2:
            a_lat_cmd = a_lat_cmd / a_lat_mag * self.cfg.accel_max_lat_mps2
            a_lat_mag = self.cfg.accel_max_lat_mps2

        # ── Propulsione (fase boost) ───────────────────────────────────
        t_since_launch = t - self._t_launch
        v_mag = float(np.linalg.norm(self.vel))

        if t_since_launch <= self.cfg.burn_time_s and v_mag > 0.1:
            v_hat = self.vel / v_mag
            a_thrust: np.ndarray = (self.cfg.thrust_N / self.cfg.mass_kg) * v_hat
        else:
            a_thrust = np.zeros(3)

        # ── Gravità reale ──────────────────────────────────────────────
        a_gravity = np.array([0.0, 0.0, -G])

        # ── Accelerazione totale e integrazione ───────────────────────
        a_total = a_thrust + a_lat_cmd + a_gravity
        self.vel = self.vel + a_total * dt

        # Clipping velocità massima
        speed = float(np.linalg.norm(self.vel))
        if speed > self.cfg.speed_max_mps:
            self.vel = self.vel / speed * self.cfg.speed_max_mps

        self.pos = self.pos + self.vel * dt

        # ── Controllo quota ────────────────────────────────────────────
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0
            self.active = False
            logger.warning("[Missile] Ha colpito il suolo — mancato!")

        # Storie
        self.pos_history.append(self.pos.copy())
        self.vel_history.append(self.vel.copy())
        self.accel_lat_history.append(a_lat_mag)
        self.closing_vel_history.append(Vc)

        return self.pos

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))


# ══════════════════════════════════════════════════════════════════════════════
# Orchestratore
# ══════════════════════════════════════════════════════════════════════════════

class InterceptionSimulation:
    """
    Orchestratore della simulazione di intercettazione.

    Esempio d'uso::

        tcfg = TargetConfig(nz_max=7, speed_mps=300)
        mcfg = MissileConfig(nav_constant=4.5)
        scfg = SimConfig(seed=42)

        sim = InterceptionSimulation(tcfg, mcfg, scfg)
        result = sim.run()
        sim.plot()
    """

    def __init__(
        self,
        target_cfg: TargetConfig,
        missile_cfg: MissileConfig,
        sim_cfg: SimConfig,
    ) -> None:
        self.tcfg = target_cfg
        self.mcfg = missile_cfg
        self.scfg = sim_cfg

        rng = np.random.default_rng(sim_cfg.seed)
        self.target = Target(target_cfg, rng)
        self.missile = Missile(missile_cfg)

        self.t: float = 0.0
        self.times: list[float] = [0.0]
        self.miss_distances: list[float] = []
        self.result: Optional[dict] = None

    def run(self) -> dict:
        """
        Esegue la simulazione fino a intercettazione, fuori campo o timeout.

        Returns:
            dict con chiavi:
                intercepted      : bool
                time_s           : float — tempo di intercettazione o timeout
                miss_distance_m  : float — distanza minima di avvicinamento
                intercept_pos    : np.ndarray | None
                missile_speed_mps: float (solo se intercepted)
        """
        dt = self.scfg.dt

        # Lancio immediato verso posizione corrente del bersaglio
        self.missile.launch(self.target.pos, t0=0.0)

        # Distanza iniziale
        self.miss_distances.append(
            float(np.linalg.norm(self.target.pos - self.missile.pos))
        )

        while self.t < self.scfg.t_max and self.missile.active:
            self.t = round(self.t + dt, 6)
            self.times.append(self.t)

            # Step bersaglio
            t_pos = self.target.step(self.t, dt)
            t_vel = self.target.vel.copy()

            # Step missile
            m_pos = self.missile.step(self.t, dt, t_pos, t_vel)

            # Distanza corrente
            dist = float(np.linalg.norm(t_pos - m_pos))
            self.miss_distances.append(dist)

            # Controllo intercettazione
            if dist < self.scfg.kill_radius_m:
                logger.info(
                    f"[Sim] ✓ INTERCETTATO  t={self.t:.2f}s  "
                    f"distanza={dist:.1f}m  quota={m_pos[2]:.0f}m  "
                    f"vel_missile={self.missile.speed:.0f}m/s"
                )
                self.result = {
                    "intercepted": True,
                    "time_s": self.t,
                    "miss_distance_m": dist,
                    "intercept_pos": m_pos.copy(),
                    "missile_speed_mps": self.missile.speed,
                    "min_distance_m": min(self.miss_distances),
                }
                return self.result

        # Mancato
        min_d = min(self.miss_distances)
        logger.info(
            f"[Sim] ✗ MANCATO — avvicinamento minimo: {min_d:.0f}m "
            f"| distanza finale: {float(np.linalg.norm(self.target.pos - self.missile.pos)):.0f}m"
        )
        self.result = {
            "intercepted": False,
            "time_s": self.t,
            "miss_distance_m": min_d,
            "intercept_pos": None,
        }
        return self.result

    def plot(self, show: bool = True) -> plt.Figure:
        """Genera la visualizzazione multi-panel dell'ingaggio."""
        return Visualizer.plot(self, show=show)

    def summary(self) -> str:
        """Stringa riassuntiva del risultato."""
        if self.result is None:
            return "Simulazione non ancora eseguita."
        r = self.result
        lines = [
            "=" * 60,
            f"  RISULTATO : {'INTERCETTATO ✓' if r['intercepted'] else 'MANCATO ✗'}",
            f"  Tempo     : {r['time_s']:.2f} s",
            f"  Dist min  : {r['miss_distance_m']:.1f} m",
        ]
        if r["intercepted"]:
            ip = r["intercept_pos"]
            lines += [
                f"  Pos. INT  : ({ip[0]/1000:.1f}, {ip[1]/1000:.1f}, {ip[2]:.0f}m)",
                f"  Vel mis.  : {r['missile_speed_mps']:.0f} m/s",
            ]
        lines += [
            "",
            f"  [Target]  massa={self.tcfg.mass_kg:.0f}kg  "
            f"V={self.tcfg.speed_mps:.0f}m/s  nz_max={self.tcfg.nz_max}",
            f"  [Missile] N'={self.mcfg.nav_constant}  "
            f"V_max={self.mcfg.speed_max_mps:.0f}m/s  "
            f"a_max={self.mcfg.accel_max_lat_mps2/G:.0f}g  "
            f"boost={self.mcfg.burn_time_s:.1f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Visualizer
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Genera i grafici multi-panel della simulazione."""

    # Palette (stile "dark terminal")
    BG_MAIN   = "#0d1117"
    BG_PANEL  = "#161b22"
    COL_GRID  = "#21262d"
    COL_SPINE = "#30363d"
    COL_TEXT  = "#e6edf3"
    COL_LABEL = "#8b949e"
    COL_TARGET  = "#f78166"   # arancione-rosso
    COL_MISSILE = "#58a6ff"   # azzurro
    COL_LAUNCH  = "#3fb950"   # verde
    COL_HIT     = "#ffa657"   # oro
    COL_MISS    = "#f85149"   # rosso acceso
    COL_WARN    = "#e3b341"   # giallo

    @classmethod
    def plot(cls, sim: InterceptionSimulation, show: bool = True) -> plt.Figure:
        """Genera figura 4-panel: 3D + altitudine + velocità + distanza."""

        result  = sim.result
        target  = sim.target
        missile = sim.missile

        # ── Allineamento array ─────────────────────────────────────────
        n = min(
            len(sim.times),
            len(target.pos_history),
            len(missile.pos_history),
            len(sim.miss_distances),
        )
        times   = np.array(sim.times[:n])
        t_pos   = np.array(target.pos_history[:n])
        m_pos   = np.array(missile.pos_history[:n])
        t_vel   = np.array(target.vel_history[:n])
        m_vel   = np.array(missile.vel_history[:n])
        miss_d  = np.array(sim.miss_distances[:n])
        a_lat   = np.array(missile.accel_lat_history[:n])
        Vc      = np.array(missile.closing_vel_history[:n])
        nz_arr  = np.array(target.nz_history[:n])

        # ── Layout figura ──────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor(cls.BG_MAIN)

        gs = gridspec.GridSpec(
            4, 3,
            figure=fig,
            left=0.04, right=0.97,
            top=0.91, bottom=0.06,
            hspace=0.55, wspace=0.38,
        )

        ax3d  = fig.add_subplot(gs[:, :2], projection="3d")
        ax_alt = fig.add_subplot(gs[0, 2])
        ax_spd = fig.add_subplot(gs[1, 2])
        ax_mis = fig.add_subplot(gs[2, 2])
        ax_nz  = fig.add_subplot(gs[3, 2])

        # Stile pannelli 2D
        for ax in [ax_alt, ax_spd, ax_mis, ax_nz]:
            ax.set_facecolor(cls.BG_PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(cls.COL_SPINE)
            ax.tick_params(colors=cls.COL_LABEL, labelsize=7)
            ax.xaxis.label.set_color(cls.COL_LABEL)
            ax.yaxis.label.set_color(cls.COL_LABEL)
            ax.title.set_color(cls.COL_TEXT)
            ax.grid(True, color=cls.COL_GRID, linewidth=0.4, linestyle="--")

        # ── 3D: Traiettorie ────────────────────────────────────────────
        ax3d.set_facecolor(cls.BG_MAIN)

        # Ombra a terra del bersaglio
        ax3d.plot(
            t_pos[:, 0] / 1000, t_pos[:, 1] / 1000,
            np.zeros(len(t_pos)),
            color=cls.COL_TARGET, lw=0.6, alpha=0.25, linestyle=":",
        )

        # Traiettoria bersaglio
        ax3d.plot(
            t_pos[:, 0] / 1000, t_pos[:, 1] / 1000, t_pos[:, 2] / 1000,
            color=cls.COL_TARGET, lw=2.0, label="Bersaglio", alpha=0.9,
        )

        # Traiettoria missile
        ax3d.plot(
            m_pos[:, 0] / 1000, m_pos[:, 1] / 1000, m_pos[:, 2] / 1000,
            color=cls.COL_MISSILE, lw=2.0, label="Missile", alpha=0.9,
        )

        # Marker inizio bersaglio
        ax3d.scatter(
            t_pos[0, 0] / 1000, t_pos[0, 1] / 1000, t_pos[0, 2] / 1000,
            color=cls.COL_TARGET, s=80, marker="^", zorder=6,
            label="Bersaglio t₀",
        )

        # Sito di lancio
        ax3d.scatter(0, 0, 0, color=cls.COL_LAUNCH, s=120, marker="D",
                     zorder=7, label="Lancio")

        # Punto di intercettazione o posizione finale
        if result and result["intercepted"]:
            ip = result["intercept_pos"] / 1000
            ax3d.scatter(
                *ip, color=cls.COL_HIT, s=200, marker="*",
                zorder=8, label=f"Intercept t={result['time_s']:.1f}s",
            )
            # Linea verticale al punto di intercettazione
            ax3d.plot(
                [ip[0], ip[0]], [ip[1], ip[1]], [0, ip[2]],
                color=cls.COL_HIT, lw=0.8, linestyle="--", alpha=0.5,
            )

        ax3d.set_xlabel("X [km]", color=cls.COL_LABEL, fontsize=9, labelpad=6)
        ax3d.set_ylabel("Y [km]", color=cls.COL_LABEL, fontsize=9, labelpad=6)
        ax3d.set_zlabel("Alt [km]", color=cls.COL_LABEL, fontsize=9, labelpad=6)
        ax3d.tick_params(colors=cls.COL_LABEL, labelsize=7)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor(cls.COL_GRID)
        ax3d.yaxis.pane.set_edgecolor(cls.COL_GRID)
        ax3d.zaxis.pane.set_edgecolor(cls.COL_GRID)
        ax3d.grid(True, color=cls.COL_GRID, linewidth=0.4)

        leg = ax3d.legend(
            loc="upper left", fontsize=8,
            facecolor=cls.BG_PANEL, edgecolor=cls.COL_SPINE,
            labelcolor=cls.COL_TEXT,
        )

        status_color = cls.COL_LAUNCH if (result and result["intercepted"]) else cls.COL_MISS
        status_txt   = "INTERCETTATO ✓" if (result and result["intercepted"]) else "MANCATO ✗"
        ax3d.set_title(
            f"Ingaggio 3D — {status_txt}",
            color=status_color, fontsize=13, pad=10, fontweight="bold",
        )

        # ── Panel: Altitudine ──────────────────────────────────────────
        ax_alt.plot(times, t_pos[:, 2] / 1000, color=cls.COL_TARGET, lw=1.5,
                    label="Bersaglio")
        ax_alt.plot(times, m_pos[:, 2] / 1000, color=cls.COL_MISSILE, lw=1.5,
                    label="Missile")
        ax_alt.axhline(sim.tcfg.altitude_min_m / 1000, color=cls.COL_GRID,
                       lw=0.8, linestyle="--")
        ax_alt.axhline(sim.tcfg.altitude_max_m / 1000, color=cls.COL_GRID,
                       lw=0.8, linestyle="--")
        ax_alt.set_title("Quota", fontsize=9)
        ax_alt.set_ylabel("Alt [km]", fontsize=8)
        ax_alt.legend(fontsize=7, facecolor=cls.BG_PANEL,
                      edgecolor=cls.COL_SPINE, labelcolor=cls.COL_TEXT)

        # ── Panel: Velocità ────────────────────────────────────────────
        t_spd = np.linalg.norm(t_vel, axis=1)
        m_spd = np.linalg.norm(m_vel, axis=1)
        n2 = min(len(times), len(t_spd), len(m_spd))
        ax_spd.plot(times[:n2], t_spd[:n2], color=cls.COL_TARGET, lw=1.5,
                    label="Bersaglio")
        ax_spd.plot(times[:n2], m_spd[:n2], color=cls.COL_MISSILE, lw=1.5,
                    label="Missile")
        ax_spd.axhline(sim.mcfg.speed_max_mps, color=cls.COL_MISSILE,
                       lw=0.7, linestyle="--", alpha=0.5)
        ax_spd.set_title("Velocità", fontsize=9)
        ax_spd.set_ylabel("V [m/s]", fontsize=8)
        ax_spd.legend(fontsize=7, facecolor=cls.BG_PANEL,
                      edgecolor=cls.COL_SPINE, labelcolor=cls.COL_TEXT)

        # ── Panel: Distanza di mancanza ────────────────────────────────
        ax_mis.semilogy(times, miss_d, color=cls.COL_WARN, lw=1.5)
        ax_mis.axhline(sim.scfg.kill_radius_m, color=cls.COL_MISS,
                       lw=1.2, linestyle="--",
                       label=f"Kill radius ({sim.scfg.kill_radius_m:.0f}m)")
        ax_mis.set_title("Distanza Missile–Bersaglio", fontsize=9)
        ax_mis.set_ylabel("Distanza [m]", fontsize=8)
        ax_mis.legend(fontsize=7, facecolor=cls.BG_PANEL,
                      edgecolor=cls.COL_SPINE, labelcolor=cls.COL_TEXT)

        # ── Panel: nz bersaglio ────────────────────────────────────────
        ax_nz.plot(times[:len(nz_arr)], nz_arr[:len(times)],
                   color=cls.COL_TARGET, lw=1.2)
        ax_nz.axhline(sim.tcfg.nz_max, color=cls.COL_MISS,
                      lw=0.9, linestyle="--",
                      label=f"nz_max = {sim.tcfg.nz_max}")
        ax_nz.set_title("Fattore di Carico Bersaglio", fontsize=9)
        ax_nz.set_ylabel("nz [-]", fontsize=8)
        ax_nz.set_xlabel("Tempo [s]", fontsize=8)
        ax_nz.legend(fontsize=7, facecolor=cls.BG_PANEL,
                     edgecolor=cls.COL_SPINE, labelcolor=cls.COL_TEXT)

        # ── Super-title con parametri chiave ───────────────────────────
        stats = (
            f"Bersaglio: {sim.tcfg.mass_kg/1000:.0f}t · "
            f"V={sim.tcfg.speed_mps:.0f}m/s · nz_max={sim.tcfg.nz_max}g   "
            f"│   Missile: N'={sim.mcfg.nav_constant} · "
            f"V_max={sim.mcfg.speed_max_mps:.0f}m/s · "
            f"a_max={sim.mcfg.accel_max_lat_mps2/G:.0f}g · "
            f"boost={sim.mcfg.burn_time_s:.0f}s"
        )
        fig.suptitle(stats, fontsize=9, color=cls.COL_LABEL, y=0.97)

        if sim.scfg.save_fig:
            fig.savefig(sim.scfg.save_fig, dpi=150, bbox_inches="tight",
                        facecolor=cls.BG_MAIN)
            logger.info(f"[Visualizer] Figura salvata in '{sim.scfg.save_fig}'")

        if show:
            plt.show()

        return fig


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulazione 3D di intercettazione missilistica",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Target
    tg = p.add_argument_group("Bersaglio (Target)")
    tg.add_argument("--target-mass",    type=float, default=15000.0,
                    metavar="kg",   help="Massa del bersaglio [kg]")
    tg.add_argument("--target-speed",   type=float, default=250.0,
                    metavar="m/s",  help="Velocità di crociera [m/s]")
    tg.add_argument("--nz-max",         type=float, default=5.0,
                    metavar="g",    help="Fattore di carico max strutturale [-]")
    tg.add_argument("--alt-min",        type=float, default=3000.0,
                    metavar="m",    help="Quota minima [m AGL]")
    tg.add_argument("--alt-max",        type=float, default=12000.0,
                    metavar="m",    help="Quota massima [m AGL]")
    tg.add_argument("--maneuver-period", type=float, default=20.0,
                    metavar="s",    help="Periodo medio tra manovre [s]")
    tg.add_argument("--spawn-range",    type=float, default=40.0,
                    metavar="km",   help="Range max comparsa bersaglio [km]")

    # Missile
    mg = p.add_argument_group("Missile")
    mg.add_argument("--missile-mass",   type=float, default=300.0,
                    metavar="kg",   help="Massa missile [kg]")
    mg.add_argument("--thrust",         type=float, default=60000.0,
                    metavar="N",    help="Spinta boost [N]")
    mg.add_argument("--burn-time",      type=float, default=5.0,
                    metavar="s",    help="Durata combustione [s]")
    mg.add_argument("--missile-speed",  type=float, default=900.0,
                    metavar="m/s",  help="Velocità massima missile [m/s]")
    mg.add_argument("--accel-max",      type=float, default=300.0,
                    metavar="m/s2", help="Accelerazione laterale max [m/s²]")
    mg.add_argument("--nav-constant",   type=float, default=4.0,
                    metavar="N'",   help="Costante di navigazione PN N'")

    # Simulazione
    sg = p.add_argument_group("Simulazione")
    sg.add_argument("--dt",             type=float, default=0.05,
                    metavar="s",    help="Passo temporale [s]")
    sg.add_argument("--t-max",          type=float, default=120.0,
                    metavar="s",    help="Tempo massimo simulazione [s]")
    sg.add_argument("--kill-radius",    type=float, default=30.0,
                    metavar="m",    help="Raggio di kill [m]")
    sg.add_argument("--seed",           type=int,   default=None,
                    help="Seed casuale (None = casuale)")
    sg.add_argument("--no-plot",        action="store_true",
                    help="Non mostrare i grafici")
    sg.add_argument("--save",           type=str,   default=None,
                    metavar="FILE", help="Salva figura in FILE (es. sim.png)")
    sg.add_argument("--runs",           type=int,   default=1,
                    metavar="N",    help="Esegui N run (statistiche multi-run)")

    return p


def single_run(args: argparse.Namespace) -> dict:
    """Esegui un singolo run e ritorna il risultato."""
    tcfg = TargetConfig(
        mass_kg             = args.target_mass,
        speed_mps           = args.target_speed,
        nz_max              = args.nz_max,
        altitude_min_m      = args.alt_min,
        altitude_max_m      = args.alt_max,
        maneuver_period_s   = args.maneuver_period,
        spawn_range_km      = args.spawn_range,
    )
    mcfg = MissileConfig(
        mass_kg             = args.missile_mass,
        thrust_N            = args.thrust,
        burn_time_s         = args.burn_time,
        speed_max_mps       = args.missile_speed,
        accel_max_lat_mps2  = args.accel_max,
        nav_constant        = args.nav_constant,
    )
    scfg = SimConfig(
        dt           = args.dt,
        t_max        = args.t_max,
        kill_radius_m= args.kill_radius,
        seed         = args.seed,
        save_fig     = args.save,
    )

    sim = InterceptionSimulation(tcfg, mcfg, scfg)
    result = sim.run()

    print(sim.summary())

    if not args.no_plot:
        sim.plot(show=True)

    return result


def multi_run(args: argparse.Namespace) -> None:
    """Esegui N run con seed incrementali e stampa statistiche aggregate."""
    import copy
    intercepted = 0
    times: list[float] = []
    min_dists: list[float] = []

    base_seed = args.seed if args.seed is not None else 0

    logger.info(f"\n{'─'*50}")
    logger.info(f"Multi-run: {args.runs} simulazioni")
    logger.info(f"{'─'*50}")

    for i in range(args.runs):
        args_i = copy.copy(args)
        args_i.seed = base_seed + i
        args_i.no_plot = True
        args_i.save = None

        tcfg = TargetConfig(
            mass_kg=args.target_mass, speed_mps=args.target_speed,
            nz_max=args.nz_max, altitude_min_m=args.alt_min,
            altitude_max_m=args.alt_max, maneuver_period_s=args.maneuver_period,
            spawn_range_km=args.spawn_range,
        )
        mcfg = MissileConfig(
            mass_kg=args.missile_mass, thrust_N=args.thrust,
            burn_time_s=args.burn_time, speed_max_mps=args.missile_speed,
            accel_max_lat_mps2=args.accel_max, nav_constant=args.nav_constant,
        )
        scfg = SimConfig(
            dt=args.dt, t_max=args.t_max,
            kill_radius_m=args.kill_radius, seed=base_seed + i,
        )
        sim = InterceptionSimulation(tcfg, mcfg, scfg)
        r = sim.run()

        if r["intercepted"]:
            intercepted += 1
            times.append(r["time_s"])
        min_dists.append(r["miss_distance_m"])
        print(f"  Run {i+1:3d}: {'✓' if r['intercepted'] else '✗'}  "
              f"t={r['time_s']:.1f}s  min_dist={r['miss_distance_m']:.0f}m")

    n = args.runs
    print(f"\n{'═'*50}")
    print(f"  Pk (kill probability) : {intercepted}/{n} = {intercepted/n*100:.1f}%")
    if times:
        print(f"  Tempo medio interc.   : {np.mean(times):.1f} s ± {np.std(times):.1f}")
    print(f"  Dist min media        : {np.mean(min_dists):.0f} m")
    print(f"{'═'*50}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.runs > 1:
        multi_run(args)
    else:
        single_run(args)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
