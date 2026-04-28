import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from PYTHON_ANALYSIS.cellgpu_analysis import (
    compute_metrics,
    compute_msd,
    load_nc_file,
    parse_params_from_filename,
    resolve_inputs,
    run_metrics,
    write_ensemble_txts,
)
from PYTHON_ANALYSIS.example_data import (
    create_avm_example,
    create_example_nc,
    create_spv_example,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic nc builders
# ---------------------------------------------------------------------------

def _make_migrating_nc(path: Path, n_bg: int = 10, t_len: int = 20,
                        dx_active: float = 0.5, dx_passive: float = 0.3,
                        lx: float = 200.0, ly: float = 200.0) -> None:
    """Synthetic SPV nc where active cell leads in -x (leftward migration, as in the real sim).

    Active cell (type 1): starts at x=150, moves left at dx_active per frame.
    Passive cell (type 2): starts at x=160 (behind active toward the right), moves left slower.
    Active cell is further left → polarity > 0.5.
    Box is large (200×200 default) so cells never wrap during t_len frames.
    Director for active cell points in -x direction (angle = π).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cells = n_bg + 2
    rng = np.random.default_rng(42)
    bg_x = rng.uniform(1.0, min(lx * 0.3, lx - 1.0), n_bg)  # background on left side
    bg_y = rng.uniform(1.0, ly - 1.0, n_bg)

    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("time", t_len)
        nc.createDimension("coord", 2 * n_cells)
        nc.createDimension("cell", n_cells)
        nc.createDimension("boxcoord", 4)

        tv = nc.createVariable("time",      "f8", ("time",))
        pv = nc.createVariable("pos",       "f8", ("time", "coord"))
        yv = nc.createVariable("type",      "i4", ("time", "cell"))
        bv = nc.createVariable("BoxMatrix", "f8", ("boxcoord", "time"))
        dv = nc.createVariable("director",  "f8", ("cell", "time"))

        tv[:] = np.arange(t_len, dtype=float) * 0.1

        pos = np.zeros((t_len, 2 * n_cells), dtype=float)
        for t in range(t_len):
            # Active cell: starts at x=150 (right), moves left faster
            pos[t, 0] = 150.0 - dx_active * t
            pos[t, 1] = 100.0
            # Passive cell: starts at x=160 (even further right, behind active),
            # moves left slower — active remains in front (smaller x)
            pos[t, 2] = 160.0 - dx_passive * t
            pos[t, 3] = 100.0
            # Background cells: fixed on left side
            pos[t, 4::2] = bg_x
            pos[t, 5::2] = bg_y
        pv[:] = pos

        ctype = np.zeros((t_len, n_cells), dtype=np.int32)
        ctype[:, 0] = 1   # active
        ctype[:, 1] = 2   # passive
        yv[:] = ctype

        box = np.zeros((4, t_len), dtype=float)
        box[0, :] = lx
        box[3, :] = ly
        bv[:] = box

        dirs = np.zeros((n_cells, t_len), dtype=float)
        dirs[0, :] = np.pi   # active cell points -x (leftward migration)
        dv[:] = dirs


def _make_nc(path: Path, n_cells: int = 4, t_len: int = 6) -> None:
    """Minimal synthetic nc — two migratory cells, static."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("time", t_len)
        nc.createDimension("coord", 2 * n_cells)
        nc.createDimension("cell", n_cells)
        nc.createDimension("boxcoord", 4)

        tv = nc.createVariable("time",      "f8", ("time",))
        pv = nc.createVariable("pos",       "f8", ("time", "coord"))
        yv = nc.createVariable("type",      "i4", ("time", "cell"))
        bv = nc.createVariable("BoxMatrix", "f8", ("boxcoord", "time"))
        dv = nc.createVariable("director",  "f8", ("cell", "time"))

        tv[:] = np.arange(t_len, dtype=float) * 0.1

        pos = np.zeros((t_len, 2 * n_cells), dtype=float)
        for t in range(t_len):
            xv = np.array([2.0 + 0.2 * t, 2.2 + 0.2 * t, 6.0, 7.0], dtype=float)
            yv_ = np.array([3.0, 3.1, 6.0, 7.0], dtype=float)
            pos[t, 0::2] = xv
            pos[t, 1::2] = yv_
        pv[:] = pos

        ctype = np.zeros((t_len, n_cells), dtype=np.int32)
        ctype[:, 0] = 1
        ctype[:, 1] = 2
        yv[:] = ctype

        box = np.zeros((4, t_len), dtype=float)
        box[0, :] = 10.0
        box[3, :] = 10.0
        bv[:] = box

        dirs = np.zeros((n_cells, t_len), dtype=float)
        dv[:] = dirs


def _spv_nc_path(td: str) -> Path:
    return (
        Path(td) / "DATA" / "POS_GAMMA_HETERO"
        / "Data_t_1000000_N_400_dt_0.0001_Ka_10_Kp_1_A0_1-1"
          "_P0_3.7-3.7_s0_3.7-3.7_c_Size_8_a_Cells_4"
          "_va_2_vp_0_vb_0.01_T1_0_T2_0_T3_0"
          "_Dr_b_0.05_Dr_tp_0.005_vD_tp_0.02_IC_3_ensem_1.nc"
    )


# ---------------------------------------------------------------------------
# Load + shape tests
# ---------------------------------------------------------------------------

class TestLoadNcFile(unittest.TestCase):
    def test_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "sample.nc"
            _make_nc(p)
            data = load_nc_file(p)
            self.assertEqual(data.x.shape, (4, 6))
            self.assertEqual(data.y.shape, (4, 6))
            self.assertEqual(data.cell_type.shape, (6, 4))
            self.assertEqual(data.time.shape, (6,))
            self.assertIsNotNone(data.directors)
            self.assertEqual(data.directors.shape, (6, 4))

    def test_model_type_spv(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "sample.nc"
            _make_nc(p)
            data = load_nc_file(p)
            self.assertEqual(data.model_type, "SPV")
            self.assertIsNone(data.vx)


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

class TestParseParams(unittest.TestCase):
    def test_spv_filename(self):
        p = Path(
            "DATA/POS_GAMMA_HETERO/"
            "Data_t_1000000_N_400_dt_0.0001_Ka_10_Kp_1_A0_1-1"
            "_P0_3.7-3.7_s0_3.7-3.7_c_Size_8_a_Cells_4"
            "_va_2_vp_0_vb_0.01_T1_0_T2_0_T3_0"
            "_Dr_b_0.05_Dr_tp_0.005_vD_tp_0.02_IC_3_ensem_1.nc"
        )
        params = parse_params_from_filename(p)
        self.assertEqual(params["N"], 400)
        self.assertAlmostEqual(params["Ka"], 10.0)
        self.assertAlmostEqual(params["sa"], 3.7)
        self.assertAlmostEqual(params["T1"], 0.0)
        self.assertEqual(params["m_val"], 8)
        self.assertEqual(params["a_val"], 4)
        self.assertEqual(params["ensem"], 1)
        self.assertAlmostEqual(params["Dr_tp"], 0.005)

    def test_generic_filename_no_crash(self):
        p = Path("DATA/EXAMPLES/example_spv_small_ensem_1.nc")
        params = parse_params_from_filename(p)
        self.assertIsInstance(params, dict)


# ---------------------------------------------------------------------------
# Physics: cluster displacement direction
# ---------------------------------------------------------------------------

class TestClusterDisplacementPhysics(unittest.TestCase):
    def test_cluster_moves_in_negative_x(self):
        """Cluster COM should move in -x direction (leftward, matching real sim physics)."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "migrating.nc"
            _make_migrating_nc(p, n_bg=10, t_len=20, dx_active=0.5, dx_passive=0.3)
            data = load_nc_file(p)
            series, summary = compute_metrics(data, migratory_types=[1, 2])

            self.assertTrue(
                np.isfinite(summary["cluster_disp_x"]),
                "cluster_disp_x should be finite"
            )
            self.assertLess(
                summary["cluster_disp_x"], 0.0,
                f"Expected negative x-displacement (leftward migration), got {summary['cluster_disp_x']:.3f}"
            )

    def test_active_cell_leads_passive(self):
        """Active cell (leftmost) should have more negative displacement than passive."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "migrating.nc"
            _make_migrating_nc(p, n_bg=10, t_len=20, dx_active=0.5, dx_passive=0.3)
            data = load_nc_file(p)
            _, summary = compute_metrics(data, migratory_types=[1, 2])

            disp_active  = summary["mean_disp_x_active"]
            disp_passive = summary["mean_disp_x_passive"]
            self.assertTrue(np.isfinite(disp_active),  "active displacement should be finite")
            self.assertTrue(np.isfinite(disp_passive), "passive displacement should be finite")
            # Active moves left faster → more negative displacement
            self.assertLess(
                disp_active, disp_passive,
                f"Active ({disp_active:.3f}) should displace further left than passive ({disp_passive:.3f})"
            )

    def test_com_x_decreases_monotonically(self):
        """COM x should decrease each frame (cells drift leftward)."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "migrating.nc"
            _make_migrating_nc(p, n_bg=0, t_len=20, dx_active=0.4, dx_passive=0.2)
            data = load_nc_file(p)
            series, _ = compute_metrics(data, migratory_types=[1, 2])

            com_x = series["com_x"]
            valid = com_x[np.isfinite(com_x)]
            self.assertGreater(len(valid), 2)
            diffs = np.diff(valid)
            n_increasing = np.sum(diffs > 1e-9)
            self.assertEqual(n_increasing, 0,
                             f"COM x increased at {n_increasing} frames — should monotonically decrease")


# ---------------------------------------------------------------------------
# Physics: speed, polarity, MSD
# ---------------------------------------------------------------------------

class TestSpeedAndPolarity(unittest.TestCase):
    def test_nonzero_speed(self):
        """Cluster speed should be > 0 when cells are moving."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "migrating.nc"
            _make_migrating_nc(p, n_bg=5, t_len=15, dx_active=0.3, dx_passive=0.2)
            data = load_nc_file(p)
            series, summary = compute_metrics(data, migratory_types=[1, 2])

            spd = series["speed"]
            valid_spd = spd[np.isfinite(spd)]
            self.assertGreater(len(valid_spd), 0)
            self.assertGreater(np.mean(valid_spd), 0.0, "Mean speed should be > 0")
            self.assertGreater(summary["cluster_speed_mean"], 0.0)

    def test_polarity_in_range(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "sample.nc"
            _make_nc(p)
            data = load_nc_file(p)
            series, _ = compute_metrics(data, migratory_types=[1, 2])
            pol = series["polarity"]
            valid = pol[np.isfinite(pol)]
            if len(valid):
                self.assertTrue(np.all(valid >= 0.0) and np.all(valid <= 1.0))

    def test_polarity_above_half_when_active_leads(self):
        """When active cell is clearly in front of passive, polarity should exceed 0.5."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "migrating.nc"
            # Large dx difference so active clearly leads throughout; large box to avoid wrap
            _make_migrating_nc(p, n_bg=5, t_len=15, dx_active=1.0, dx_passive=0.1,
                                lx=500.0, ly=500.0)
            data = load_nc_file(p)
            series, summary = compute_metrics(data, migratory_types=[1, 2])

            pol = series["polarity"]
            valid = pol[np.isfinite(pol) & (pol > 0)]
            if len(valid):
                self.assertGreater(
                    np.mean(valid), 0.5,
                    f"Mean polarity {np.mean(valid):.3f} should be > 0.5 when active leads"
                )


class TestMSD(unittest.TestCase):
    def test_msd_shape_and_values(self):
        rng = np.random.default_rng(42)
        x = rng.random((3, 10))
        y = rng.random((3, 10))
        msd = compute_msd(x, y)
        self.assertEqual(msd.shape, (3, 10))
        self.assertTrue(np.all(msd[:, 0] == 0.0))
        self.assertTrue(np.all(msd[:, 1:] >= 0.0))

    def test_msd_grows_with_time_for_drifting_cells(self):
        """MSD should increase over time for cells with net drift."""
        t_len = 20
        # Two cells drifting at constant velocity
        x = np.zeros((2, t_len))
        y = np.zeros((2, t_len))
        for t in range(t_len):
            x[:, t] = 0.5 * t   # linear drift
        msd = compute_msd(x, y)
        mean_msd = np.mean(msd, axis=0)
        # MSD at delta=10 should be greater than at delta=1
        self.assertGreater(mean_msd[10], mean_msd[1])

    def test_msd_non_negative(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "sample.nc"
            _make_nc(p)
            data = load_nc_file(p)
            series, _ = compute_metrics(data, migratory_types=[1, 2])
            msd = series["msd_mean"]
            self.assertTrue(np.all(np.isnan(msd) | (msd >= 0)))


# ---------------------------------------------------------------------------
# Parameter propagation
# ---------------------------------------------------------------------------

class TestFilenameParams(unittest.TestCase):
    def test_filename_params_propagate(self):
        with tempfile.TemporaryDirectory() as td:
            p = _spv_nc_path(td)
            _make_nc(p, n_cells=4, t_len=6)
            data = load_nc_file(p)
            self.assertEqual(data.params.get("N"), 400)
            self.assertEqual(data.params.get("a_val"), 4)
            self.assertEqual(data.params.get("m_val"), 8)


# ---------------------------------------------------------------------------
# I/O: resolve_inputs, run_metrics, write_ensemble_txts
# ---------------------------------------------------------------------------

class TestResolveInputs(unittest.TestCase):
    def test_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "DATA" / "POS_GAMMA_HETERO"
            _make_nc(root / "run_ensem_1.nc")
            _make_nc(root / "run_ensem_2.nc")
            result = resolve_inputs(str(root / "*.nc"), ensemble_count=1)
            self.assertEqual(len(result), 2)

    def test_ensemble_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "DATA" / "POS_GAMMA_HETERO"
            _make_nc(root / "run_ensem_1.nc")
            _make_nc(root / "run_ensem_2.nc")
            result = resolve_inputs(str(root / "run_ensem_{i}.nc"), ensemble_count=5)
            self.assertEqual(len(result), 2)

    def test_missing_returns_empty(self):
        result = resolve_inputs("/nonexistent/path/*.nc", ensemble_count=1)
        self.assertEqual(result, [])


class TestRunMetrics(unittest.TestCase):
    def test_writes_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "run_ensem_1.nc"
            _make_nc(p)
            out = Path(td) / "PROCESSED_DATA"
            args = argparse.Namespace(
                input=str(p), output_dir=str(out),
                ensemble_count=100, migratory_types="1,2",
                voronoi=False, t1=False,
            )
            code = run_metrics(args)
            self.assertEqual(code, 0)
            self.assertTrue((out / "ensemble_summary.csv").exists())
            self.assertTrue((out / "ensemble_aggregates.json").exists())
            scenario_dir = out / "POS_GAMMA_HETERO"
            self.assertTrue(any(scenario_dir.glob("*_timeseries.csv")))
            self.assertTrue(any(scenario_dir.glob("*_summary.json")))

    def test_aggregates_has_sem(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "DATA" / "POS_GAMMA_HETERO"
            for i in range(1, 4):
                _make_nc(root / f"run_ensem_{i}.nc")
            out = Path(td) / "PROCESSED_DATA"
            args = argparse.Namespace(
                input=str(root / "run_ensem_{i}.nc"), output_dir=str(out),
                ensemble_count=5, migratory_types="1,2",
                voronoi=False, t1=False,
            )
            run_metrics(args)
            with (out / "ensemble_aggregates.json").open() as f:
                aggs = json.load(f)
            for key, entry in aggs.items():
                if isinstance(entry, dict) and "sem" in entry:
                    self.assertGreaterEqual(entry["sem"], 0.0)
                    break

    def test_displacement_sign_in_csv(self):
        """run_metrics should produce a negative cluster_disp_x for leftward migration."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "migrating_ensem_1.nc"
            _make_migrating_nc(p, n_bg=5, t_len=15)
            out = Path(td) / "PROCESSED_DATA"
            args = argparse.Namespace(
                input=str(p), output_dir=str(out),
                ensemble_count=1, migratory_types="1,2",
                voronoi=False, t1=False,
            )
            run_metrics(args)
            with (out / "ensemble_aggregates.json").open() as f:
                aggs = json.load(f)
            self.assertIn("cluster_disp_x", aggs)
            self.assertLess(aggs["cluster_disp_x"]["mean"], 0.0,
                            "Leftward migrating cluster should have negative disp_x in aggregates")


class TestWriteEnsembleTxts(unittest.TestCase):
    def test_writes_dist_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "DATA" / "POS_GAMMA_HETERO" / "run"
            summaries = [
                {"cluster_disp_x": 1.5, "closeness": 0.3},
                {"cluster_disp_x": 2.0, "closeness": 0.4},
            ]
            timeseries = [
                {"speed": np.array([0.0, 0.1, 0.2]), "msd_mean": np.array([0.0, 0.01, 0.04])},
                {"speed": np.array([0.0, 0.15, 0.25]), "msd_mean": np.array([0.0, 0.02, 0.06])},
            ]
            write_ensemble_txts(base, summaries, timeseries)
            dist_file = Path(str(base) + "_ensem_cluster_dist_x.txt")
            self.assertTrue(dist_file.exists())
            content = dist_file.read_text().strip()
            vals = [float(x) for x in content.split()]
            self.assertAlmostEqual(vals[0], 1.5, places=4)
            self.assertAlmostEqual(vals[1], 2.0, places=4)


# ---------------------------------------------------------------------------
# Example generators + SPV/AVM model detection
# ---------------------------------------------------------------------------

class TestExampleGenerators(unittest.TestCase):
    def test_generic_example(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATA" / "EXAMPLES" / "example_spv_small_ensem_1.nc"
            create_example_nc(p, n_cells=7, t_len=11)
            self.assertTrue(p.exists())
            data = load_nc_file(p)
            self.assertEqual(data.x.shape, (7, 11))

    def test_spv_example_filename_and_params(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "DATA" / "POS_GAMMA_HETERO"
            path = create_spv_example(out_dir, N=400, m_val=8, a_val=4,
                                      ensem=1, t_steps=200, overwrite=True)
            self.assertTrue(path.exists())
            self.assertIn("N_400", path.name)
            self.assertIn("c_Size_8", path.name)
            self.assertIn("a_Cells_4", path.name)
            self.assertIn("ensem_1", path.name)
            data = load_nc_file(path)
            self.assertEqual(data.model_type, "SPV")
            self.assertEqual(data.params.get("N"), 400)
            self.assertEqual(data.params.get("m_val"), 8)
            self.assertEqual(data.params.get("a_val"), 4)

    def test_avm_example_filename_and_params(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "DATA" / "POS_GAMMA_VERTEX_HETERO"
            path = create_avm_example(out_dir, N=400, m_val=8, a_val=4,
                                      ensem=1, t_steps=200, overwrite=True)
            self.assertTrue(path.exists())
            self.assertIn("N_400", path.name)
            self.assertIn("ensem_1", path.name)
            data = load_nc_file(path)
            self.assertEqual(data.model_type, "AVM")
            self.assertEqual(data.params.get("N"), 400)
            # AVM should have vertex arrays
            self.assertIsNotNone(data.vx)
            self.assertIsNotNone(data.vy)

    def test_spv_and_avm_metrics_run(self):
        """compute_metrics must not crash on either model type."""
        with tempfile.TemporaryDirectory() as td:
            spv_dir = Path(td) / "DATA" / "POS_GAMMA_HETERO"
            avm_dir = Path(td) / "DATA" / "POS_GAMMA_VERTEX_HETERO"
            spv_path = create_spv_example(spv_dir, N=400, m_val=8, a_val=4,
                                          ensem=1, t_steps=50, overwrite=True)
            avm_path = create_avm_example(avm_dir, N=400, m_val=8, a_val=4,
                                          ensem=1, t_steps=50, overwrite=True)
            for path in [spv_path, avm_path]:
                data = load_nc_file(path)
                series, summary = compute_metrics(data, migratory_types=[1, 2])
                self.assertIn("cluster_disp_x", summary)
                self.assertIn("polarity_mean", summary)
                self.assertIn("msd_mean", series)
                self.assertEqual(len(series["time"]), data.x.shape[1])


if __name__ == "__main__":
    unittest.main()
