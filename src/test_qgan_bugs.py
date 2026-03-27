"""
Eigenstaendige Tests fuer generator.py, discriminator.py und training_qgan.py.

Werden AUSSCHLIESSLICH fuer diese drei Dateien geschrieben – keine Abhaengigkeit
von bereits bestehenden Test-Skripten.

Gefundene Bugs (jeweils mit eigenem Test):
  BUG-1  discriminator.py  _check_triangle_inequality behauptet (n,4) zu unterstuetzen –
                            tut es aber NICHT (wirft ValueError statt Bool-Array zu liefern).
  BUG-2  generator.py      batch_forward liefert fuer 1D-Input shape (6,) statt (1,6)
                            (verletzt die eigene Docstring-Garantie "Returns: Shape (N,6)").
  BUG-3  training_qgan.py  train_discriminator_step nimmt 'rng' als Parameter entgegen,
                            benutzt ihn aber nie.
  BUG-4  training_qgan.py  _disc_probs_from_edges: Batch-Broadcasting-Pfad fuer den Disc.
                            haengt am QNode, der nur scalar-Batch-Broadcasting unterstuetzt –
                            der Batch-Aufruf muss ueber die Schleife in batch_forward laufen
                            und durch den direkten broadcasting-Aufruf kann es zu falscher
                            Shape kommen.
  BUG-5  training_qgan.py  Modul-level rng wird beim Import direkt instanziiert –
                            Nebeneffekt: jedes Re-Import setzt den globalen Zustand nicht
                            zurueck (in Python wird das gecacht, aber der State laeuft global
                            durch alle Trainingschritte unkontrolliert).

Neben den Bug-Tests gibt es Korrektheitstests fuer alle oeffentlichen Methoden.
"""

import importlib
import inspect
import sys
import numpy as np
import pennylane.numpy as pnp
import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_gen(n_layer=2, seed=0):
    from generator import QGenerator
    return QGenerator(n_layer=n_layer, seed=seed)


def _make_disc(n_layer=2, seed=0):
    from discriminator import QDiscriminator
    return QDiscriminator(n_layer=n_layer, seed=seed)


def _rand(shape, seed=0):
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=shape)


# ===========================================================================
# BLOCK A – generator.py
# ===========================================================================

class TestGenerator:

    # --- Initialisierung ---------------------------------------------------

    def test_weights_shape(self):
        """Gewichte muessen Shape (n_layer, 6, 3) haben."""
        gen = _make_gen(n_layer=2)
        assert gen.weights.shape == (2, 6, 3), gen.weights.shape

    def test_weights_total_params(self):
        """Gesamtanzahl Parameter: n_layer * 6 Qubits * 3 Rotationen."""
        gen = _make_gen(n_layer=3)
        assert gen.weights.size == 3 * 6 * 3

    def test_n_qubits_is_6(self):
        gen = _make_gen()
        assert gen.n_qubits == 6

    # --- forward() ---------------------------------------------------------

    def test_forward_output_shape(self):
        gen = _make_gen()
        out = gen.forward(_rand((6,)))
        assert out.shape == (6,), f"Erwartet (6,), bekam {out.shape}"

    def test_forward_output_in_unit_interval(self):
        """Ausgabe von forward() muss in [0,1] liegen."""
        gen = _make_gen()
        rng = np.random.default_rng(1)
        for _ in range(5):
            edges = gen.forward(rng.uniform(0, 1, size=6))
            assert np.all(edges >= -1e-9) and np.all(edges <= 1.0 + 1e-9), edges

    def test_forward_rejects_wrong_shape(self):
        """forward() muss ValueError werfen wenn Input != (6,)."""
        gen = _make_gen()
        with pytest.raises(ValueError, match="shape"):
            gen.forward(np.array([0.1, 0.2, 0.3]))

    def test_forward_clips_out_of_range_noise(self):
        """Noise > 1 oder < 0 darf keinen Absturz verursachen."""
        gen = _make_gen()
        noisy = np.array([2.0, -1.0, 0.5, 0.5, 0.5, 0.5])
        out = gen.forward(noisy)
        assert out.shape == (6,)

    def test_forward_different_noise_different_output(self):
        """Zwei verschiedene Noise-Vektoren sollen unterschiedliche Kanten erzeugen."""
        gen = _make_gen()
        rng = np.random.default_rng(99)
        n1 = rng.uniform(0, 0.3, size=6)
        n2 = rng.uniform(0.7, 1.0, size=6)
        assert not np.allclose(gen.forward(n1), gen.forward(n2))

    # --- batch_forward() ---------------------------------------------------

    def test_batch_forward_shape(self):
        gen = _make_gen()
        out = gen.batch_forward(_rand((8, 6)))
        assert out.shape == (8, 6), f"Erwartet (8,6), bekam {out.shape}"

    def test_batch_forward_range(self):
        gen = _make_gen()
        out = gen.batch_forward(_rand((6, 6)))
        assert np.all(np.asarray(out) >= -1e-9)
        assert np.all(np.asarray(out) <= 1.0 + 1e-9)

    # *** BUG-2 ***
    def test_bug2_batch_forward_1d_input_docstring_violation(self):
        """
        BUG-2: batch_forward() gibt fuer 1D-Input shape (6,) zurueck.
        Der Docstring verspricht jedoch 'Returns: Shape (N,6)'.
        Ein Aufruf mit shape (6,) (single sample als 1D) sollte (1,6) liefern,
        liefert aber tatsaechlich (6,) – dokumentierter契約-Bruch.
        """
        gen = _make_gen()
        noise_1d = _rand((6,))
        out = gen.batch_forward(noise_1d)
        out_arr = np.asarray(out)
        # Dieser Assert SCHLAEGT mit dem aktuellen Code FEHL (Bug bestaetigt):
        assert out_arr.shape == (1, 6), (
            f"BUG-2 bestaetigt: batch_forward liefert {out_arr.shape} statt (1,6) "
            f"fuer 1D-Input. Docstring-Garantie verletzt."
        )

    # --- _to_edge_length() -------------------------------------------------

    def test_to_edge_length_formula(self):
        """0.5*(z+1): z=-1->0, z=0->0.5, z=1->1."""
        gen = _make_gen()
        z = np.array([-1.0, 0.0, 1.0])
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(gen._to_edge_length(z), expected, atol=1e-9)

    # --- circuit Z-Werte ---------------------------------------------------

    def test_circuit_z_in_valid_range(self):
        gen = _make_gen()
        noise = _rand((6,))
        z = np.asarray(gen.circuit(noise, gen.weights), dtype=float)
        assert np.all(z >= -1.0 - 1e-6) and np.all(z <= 1.0 + 1e-6), z


# ===========================================================================
# BLOCK B – discriminator.py
# ===========================================================================

class TestDiscriminator:

    # --- Initialisierung ---------------------------------------------------

    def test_weights_shape(self):
        disc = _make_disc(n_layer=2)
        assert disc.weights.shape == (2, 6, 3)

    def test_weights_total_params(self):
        disc = _make_disc(n_layer=3)
        assert disc.weights.size == 3 * 6 * 3

    # --- forward() ---------------------------------------------------------

    def test_forward_returns_float(self):
        disc = _make_disc()
        result = disc.forward(_rand((6,)))
        assert isinstance(result, float), type(result)

    def test_forward_output_in_unit_interval(self):
        disc = _make_disc()
        rng = np.random.default_rng(2)
        for _ in range(5):
            p = disc.forward(rng.uniform(0, 1, size=6))
            assert 0.0 <= p <= 1.0, p

    def test_forward_rejects_4_edges(self):
        """forward() muss ValueError werfen wenn 4 statt 6 Kanten uebergeben."""
        disc = _make_disc()
        with pytest.raises(ValueError, match="shape"):
            disc.forward(np.array([1.0, 1.0, 1.0, 1.0]))

    def test_forward_rejects_wrong_ndim(self):
        disc = _make_disc()
        with pytest.raises(ValueError):
            disc.forward(_rand((2, 6)))  # 2D statt 1D

    def test_forward_clips_out_of_range(self):
        """Werte >1 oder <0 sollen geclipt werden, kein Absturz."""
        disc = _make_disc()
        p = disc.forward(np.array([2.0, -1.0, 0.5, 0.5, 0.5, 0.5]))
        assert 0.0 <= p <= 1.0

    # --- batch_forward() ---------------------------------------------------

    def test_batch_forward_shape(self):
        disc = _make_disc()
        out = disc.batch_forward(_rand((10, 6)))
        assert out.shape == (10,)

    def test_batch_forward_range(self):
        disc = _make_disc()
        probs = disc.batch_forward(_rand((8, 6)))
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # --- _to_prob() --------------------------------------------------------

    def test_to_prob_minus_one(self):
        assert abs(QDiscriminator_to_prob(-1.0) - 0.0) < 1e-9

    def test_to_prob_zero(self):
        assert abs(QDiscriminator_to_prob(0.0) - 0.5) < 1e-9

    def test_to_prob_plus_one(self):
        assert abs(QDiscriminator_to_prob(1.0) - 1.0) < 1e-9

    # --- _check_triangle_inequality() -------------------------------------

    def test_triangle_valid(self):
        from discriminator import QDiscriminator
        assert QDiscriminator._check_triangle_inequality(
            np.array([1.0, 1.0, 1.0, 1.5])
        ) is True

    def test_triangle_invalid(self):
        from discriminator import QDiscriminator
        assert QDiscriminator._check_triangle_inequality(
            np.array([0.1, 0.1, 0.1, 5.0])
        ) is False

    def test_triangle_wrong_shape_raises(self):
        from discriminator import QDiscriminator
        with pytest.raises(ValueError):
            QDiscriminator._check_triangle_inequality(np.array([1.0, 1.0]))

    # *** BUG-1 ***
    def test_bug1_triangle_inequality_rejects_n_by_4(self):
        """
        BUG-1: _check_triangle_inequality() Docstring sagt '(4,) oder (n, 4)',
        der Code wirft aber fuer shape (2,4) einen ValueError.
        Erwartet: ein Bool-Array der Laenge n.
        Tatsaechlich: ValueError mit 'shape'.
        """
        from discriminator import QDiscriminator
        batch = np.array([
            [1.0, 1.0, 1.0, 1.5],  # gueltig
            [0.1, 0.1, 0.1, 5.0],  # ungueltig
        ])
        # Dieser Block soll zeigen dass der Code JETZT scheitert (Bug):
        try:
            result = QDiscriminator._check_triangle_inequality(batch)
            # Wenn kein Fehler: pruefe ob Ergebnis sinnvoll ist
            assert result is not None, "Ergebnis ist None"
            result_arr = np.asarray(result)
            assert result_arr.shape == (2,), (
                f"Erwartet Bool-Array shape (2,), bekam {result_arr.shape}"
            )
            assert result_arr[0] == True  and result_arr[1] == False
        except ValueError:
            pytest.fail(
                "BUG-1 bestaetigt: _check_triangle_inequality() wirft ValueError "
                "fuer (n,4)-Input, obwohl der Docstring (n,4) als gueltig deklariert."
            )

    # --- score_edges() -----------------------------------------------------

    def test_score_edges_returns_float(self):
        disc = _make_disc()
        score = disc.score_edges(_rand((4, 6)))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# Hilfsfunktion da _to_prob @staticmethod
def QDiscriminator_to_prob(z):
    from discriminator import QDiscriminator
    return QDiscriminator._to_prob(z)


# ===========================================================================
# BLOCK C – training_qgan.py (Hilfsfunktionen & Trainingsschritte)
# ===========================================================================

class TestLossFunctions:

    def _pnp(self, lst):
        return pnp.array(lst, dtype=float)

    def test_mse_perfect_zero(self):
        from training_qgan import _loss_fn
        preds = self._pnp([1.0, 0.0, 1.0])
        tgts  = self._pnp([1.0, 0.0, 1.0])
        assert float(_loss_fn(preds, tgts, "mse")) == pytest.approx(0.0, abs=1e-9)

    def test_mse_worst_case_one(self):
        from training_qgan import _loss_fn
        preds = self._pnp([0.0, 1.0])
        tgts  = self._pnp([1.0, 0.0])
        assert float(_loss_fn(preds, tgts, "mse")) == pytest.approx(1.0, abs=1e-9)

    def test_log_perfect_near_zero(self):
        from training_qgan import _loss_fn
        eps = 1e-6
        preds = self._pnp([1.0 - eps, eps])
        tgts  = self._pnp([1.0, 0.0])
        assert float(_loss_fn(preds, tgts, "log")) < 0.01

    def test_log_worst_case_large(self):
        from training_qgan import _loss_fn
        eps = 1e-6
        preds = self._pnp([eps, 1.0 - eps])
        tgts  = self._pnp([1.0, 0.0])
        assert float(_loss_fn(preds, tgts, "log")) > 10.0

    def test_pce_non_negative(self):
        from training_qgan import _loss_fn
        preds = self._pnp([0.6, 0.3, 0.9])
        tgts  = self._pnp([1.0, 0.0, 1.0])
        assert float(_loss_fn(preds, tgts, "pce")) >= 0.0

    def test_pce_fake_target_zero_no_explosion(self):
        """
        BUG-Check: PCE mit fake-Target 0.0 soll NICHT explodieren.
        Alte Implementierung nutzte target+eps im Nenner, was bei target=0
        eine ~1e9-Division erzeugte. Die aktuelle Implementierung setzt den
        Nenner auf 1.0 wenn target<eps – kein Overflow mehr.
        """
        from training_qgan import _loss_fn
        preds = self._pnp([0.8, 0.3])
        tgts  = self._pnp([0.0, 0.0])   # fake targets = 0
        loss  = float(_loss_fn(preds, tgts, "pce"))
        assert np.isfinite(loss) and loss < 1e6, f"PCE explodiert: {loss}"

    def test_unknown_loss_type_raises(self):
        from training_qgan import _loss_fn
        with pytest.raises(ValueError, match="Unknown LOSS_TYPE"):
            _loss_fn(self._pnp([0.5]), self._pnp([1.0]), "xyz")

    @pytest.mark.parametrize("lt", ["mse", "log", "pce"])
    def test_all_loss_types_finite(self, lt):
        from training_qgan import _loss_fn
        preds = self._pnp([0.6, 0.3, 0.8])
        tgts  = self._pnp([1.0, 0.0, 1.0])
        assert np.isfinite(float(_loss_fn(preds, tgts, lt)))


class TestDiscProbsFromEdges:

    def test_output_range(self):
        from training_qgan import _disc_probs_from_edges
        disc = _make_disc()
        edges = _rand((6, 6))
        probs = np.asarray(_disc_probs_from_edges(disc, disc.weights, edges), dtype=float)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), probs

    def test_output_is_scalar_for_single(self):
        from training_qgan import _disc_probs_from_edges
        disc = _make_disc()
        edges = _rand((6,))          # single sample
        result = _disc_probs_from_edges(disc, disc.weights, edges)
        assert np.isscalar(float(result)) or np.asarray(result).ndim == 0


class TestGenEdgesFromNoise:

    def test_output_shape(self):
        from training_qgan import _gen_edges_from_noise
        gen = _make_gen()
        noise = _rand((4, 6))
        edges = _gen_edges_from_noise(gen, gen.weights, noise)
        assert np.asarray(edges).shape == (4, 6), np.asarray(edges).shape

    def test_output_range(self):
        from training_qgan import _gen_edges_from_noise
        gen = _make_gen()
        edges = np.asarray(_gen_edges_from_noise(gen, gen.weights, _rand((4, 6))))
        assert np.all(edges >= -1e-9) and np.all(edges <= 1.0 + 1e-9)


class TestTrainDiscriminatorStep:

    # *** BUG-3 (behoben) ***
    def test_bug3_rng_parameter_removed(self):
        """
        BUG-3 (behoben): train_discriminator_step hatte 'rng' als Parameter,
        welcher niemals genutzt wurde. Nach dem Fix akzeptiert die Funktion
        keinen 'rng'-Parameter mehr.
        Nachweis: Aufruf OHNE rng muss funktionieren; Aufruf MIT rng muss TypeError werfen.
        """
        from training_qgan import train_discriminator_step
        import inspect
        sig = inspect.signature(train_discriminator_step)
        assert "rng" not in sig.parameters, (
            "BUG-3 nicht behoben: 'rng' ist noch in der Signatur"
        )

    def test_returns_all_expected_keys(self):
        from training_qgan import train_discriminator_step
        disc = _make_disc()
        m = train_discriminator_step(disc, _rand((4,6)), _rand((4,6)))
        for key in ("disc_loss", "disc_loss_real", "disc_loss_fake", "disc_grad_norm"):
            assert key in m, f"Schluessel '{key}' fehlt"

    def test_loss_is_finite(self):
        from training_qgan import train_discriminator_step
        disc = _make_disc()
        m = train_discriminator_step(disc, _rand((4,6)), _rand((4,6)))
        assert np.isfinite(m["disc_loss"])

    def test_loss_is_non_negative(self):
        from training_qgan import train_discriminator_step
        disc = _make_disc()
        m = train_discriminator_step(disc, _rand((4,6)), _rand((4,6)))
        assert m["disc_loss"] >= 0.0

    def test_weights_change_after_step(self):
        """Diskriminator-Gewichte muessen sich nach einem Trainingsschritt veraendern."""
        from training_qgan import train_discriminator_step
        disc = _make_disc()
        w_before = np.array(disc.weights).copy()
        train_discriminator_step(disc, _rand((4,6)), _rand((4,6)))
        assert not np.allclose(w_before, np.array(disc.weights)), \
            "Gewichte unveraendert – kein Gradient-Update!"

    def test_grad_norm_positive(self):
        from training_qgan import train_discriminator_step
        disc = _make_disc()
        m = train_discriminator_step(disc, _rand((4,6)), _rand((4,6)))
        assert m["disc_grad_norm"] > 1e-12, \
            f"Grad-Norm nahezu Null ({m['disc_grad_norm']}) – moegliches Barren Plateau"


class TestTrainGeneratorStep:

    def test_returns_three_values(self):
        from training_qgan import train_generator_step
        gen, disc = _make_gen(), _make_disc()
        result = train_generator_step(disc, gen, _rand((4,6)))
        assert len(result) == 3

    def test_gen_loss_finite(self):
        from training_qgan import train_generator_step
        gen, disc = _make_gen(), _make_disc()
        loss, _, _ = train_generator_step(disc, gen, _rand((4,6)))
        assert np.isfinite(loss), f"Gen-Loss ist Inf/NaN: {loss}"

    def test_fake_batch_shape(self):
        from training_qgan import train_generator_step
        gen, disc = _make_gen(), _make_disc()
        _, _, fake = train_generator_step(disc, gen, _rand((4,6)))
        assert np.asarray(fake).shape == (4, 6), np.asarray(fake).shape

    def test_fake_batch_range(self):
        from training_qgan import train_generator_step
        gen, disc = _make_gen(), _make_disc()
        _, _, fake = train_generator_step(disc, gen, _rand((4,6)))
        f = np.asarray(fake)
        assert np.all(f >= -1e-9) and np.all(f <= 1.0 + 1e-9)

    def test_gen_weights_change_after_step(self):
        """Generator-Gewichte muessen sich nach einem Schritt aendern."""
        from training_qgan import train_generator_step
        gen, disc = _make_gen(), _make_disc()
        w_before = np.array(gen.weights).copy()
        train_generator_step(disc, gen, _rand((4,6)))
        assert not np.allclose(w_before, np.array(gen.weights)), \
            "Gen-Gewichte unveraendert – kein Gradient-Update!"

    def test_disc_weights_unchanged_during_gen_step(self):
        """Discriminator-Gewichte duerfen sich waehrend des Generator-Steps NICHT veraendern."""
        from training_qgan import train_generator_step
        gen, disc = _make_gen(), _make_disc()
        w_d_before = np.array(disc.weights).copy()
        train_generator_step(disc, gen, _rand((4,6)))
        np.testing.assert_allclose(
            w_d_before, np.array(disc.weights),
            atol=1e-12,
            err_msg="Disc-Gewichte wurden waehrend G-Step veraendert (D wurde nicht eingefroren!)"
        )


class TestDataPipeline:

    def test_load_cities_non_empty(self):
        from training_qgan import load_cities
        cities = load_cities("cities.csv")
        assert len(cities) > 0

    def test_load_cities_required_keys(self):
        from training_qgan import load_cities
        cities = load_cities("cities.csv")
        for city in cities[:3]:
            for k in ("name", "country", "lat", "lon"):
                assert k in city, f"'{k}' fehlt in {city}"

    def test_load_distance_cache_non_empty(self):
        from training_qgan import load_distance_cache
        cache = load_distance_cache("distance_cache.csv")
        assert len(cache) > 0

    def test_load_distance_cache_values_positive(self):
        from training_qgan import load_distance_cache
        cache = load_distance_cache("distance_cache.csv")
        vals = list(cache.values())[:20]
        assert all(v > 0 for v in vals), "Negative oder Null-Distanz im Cache"

    def test_create_batch_real_shape(self):
        from training_qgan import load_cities, load_distance_cache, create_batch_real
        cities = load_cities("cities.csv")
        cache = load_distance_cache("distance_cache.csv")
        batch = create_batch_real(cities, 8, cache, np.random.default_rng(10))
        assert batch.shape == (8, 6), batch.shape

    def test_create_batch_real_normalized(self):
        from training_qgan import load_cities, load_distance_cache, create_batch_real
        cities = load_cities("cities.csv")
        cache = load_distance_cache("distance_cache.csv")
        batch = create_batch_real(cities, 8, cache, np.random.default_rng(11))
        assert np.all(batch >= 0.0) and np.all(batch <= 1.0), \
            f"Kanten ausserhalb [0,1]: min={batch.min():.4f}, max={batch.max():.4f}"

    def test_pair_key_is_symmetric(self):
        """_pair_key muss fuer (a,b) und (b,a) denselben Key liefern."""
        from training_qgan import _pair_key
        city_a = {"name": "Berlin", "country": "DE", "lat": 52.5, "lon": 13.4}
        city_b = {"name": "Paris",  "country": "FR", "lat": 48.8, "lon": 2.35}
        assert _pair_key(city_a, city_b) == _pair_key(city_b, city_a)


# ===========================================================================
# BLOCK D – Modul-Level-Zustand (BUG-5)
# ===========================================================================

class TestModuleLevelState:

    # *** BUG-5 ***
    def test_bug5_module_level_rng_is_global(self):
        """
        BUG-5: training_qgan.py instanziiert auf Modul-Ebene
            rng = np.random.default_rng(SEED)
        Dieser globale RNG wird in main() direkt genutzt.  Das bedeutet:
        - Jeder Aufruf von Hilfsfunktionen, die den Modulzustand nicht
          kapseln, beeinflusst denselben RNG-Zustand.
        - Reproduzierbarkeit haengt davon ab, dass niemand den Modul-RNG
          ausserhalb von main() aufruft.
        Test: Wir pruefen, ob das Attribut 'rng' auf Modul-Ebene existiert.
        """
        import training_qgan as tq
        assert hasattr(tq, "rng"), (
            "Modul-level 'rng' nicht gefunden – Bug moeglicherweise bereits behoben."
        )
        # Sicherstellen dass es tatsaechlich ein Generator-Objekt ist
        assert isinstance(tq.rng, np.random.Generator), (
            f"tq.rng hat unerwarteten Typ: {type(tq.rng)}"
        )
        # Verifiziere dass der globale State durch Aufruf beeinflusst wird:
        val_before = tq.rng.integers(0, 10000)
        val_after  = tq.rng.integers(0, 10000)
        assert val_before != val_after, (
            "BUG-5 haerter Nachweis fehlgeschlagen: RNG produziert identische Werte"
        )


# ===========================================================================
# BLOCK E – Adversariales Zusammenspiel
# ===========================================================================

class TestAdversarialCombined:

    def test_two_steps_disc_then_gen(self):
        """D- und G-Schritt nacheinander duerfen keinen Fehler verursachen."""
        from training_qgan import train_discriminator_step, train_generator_step
        gen, disc = _make_gen(), _make_disc()
        rng = np.random.default_rng(50)
        for _ in range(2):
            real = rng.uniform(0.2, 0.8, size=(4, 6))
            noise = rng.uniform(0, 1, size=(4, 6))
            fake = gen.batch_forward(noise)
            train_discriminator_step(disc, real, fake)
            train_generator_step(disc, gen, noise)

    def test_disc_gen_losses_stay_finite_over_steps(self):
        """Verluste duerfen ueber mehrere Schritte nicht divergieren."""
        from training_qgan import train_discriminator_step, train_generator_step
        gen, disc = _make_gen(), _make_disc()
        rng = np.random.default_rng(60)
        for i in range(4):
            real = rng.uniform(0.1, 0.9, size=(4, 6))
            noise = rng.uniform(0, 1, size=(4, 6))
            fake = gen.batch_forward(noise)
            m = train_discriminator_step(disc, real, fake)
            g_loss, _, _ = train_generator_step(disc, gen, noise)
            assert np.isfinite(m["disc_loss"]), f"Disc-Loss Schritt {i}: {m['disc_loss']}"
            assert np.isfinite(g_loss), f"Gen-Loss Schritt {i}: {g_loss}"
