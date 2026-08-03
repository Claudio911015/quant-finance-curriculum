"""Tests de la comparación de salidas usada por tools/check_outputs_fresh.py.

La lógica es delicada: demasiado estricta y el CI falla siempre por ruido de
punto flotante entre entornos (conda/MKL contra pip/OpenBLAS); demasiado laxa
y deja pasar salidas rancias de verdad. Estos casos fijan ese equilibrio.
"""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "check_outputs_fresh",
    pathlib.Path(__file__).resolve().parent.parent / "tools" / "check_outputs_fresh.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
equivalent = _mod.equivalent


def test_ruido_de_punto_flotante_entre_entornos_es_equivalente():
    # Caso real: mismo notebook en conda (MKL) y en el runner (OpenBLAS).
    a = "  K=0.020: propio=0.175056007708  QL=0.175056007708  diff=0.00e+00"
    b = "  K=0.020: propio=0.175056053535  QL=0.175056053535  diff=0.00e+00"
    assert equivalent(a, b)


def test_salida_rancia_de_verdad_se_detecta():
    # Caso real: 08.3 publicaba parámetros que su código ya no producía.
    a = "parametros calibrados: v0=0.6008, kappa=8.2515, xi=2.9200"
    b = "parametros calibrados: v0=0.6172, kappa=8.4779, xi=3.0000"
    assert not equivalent(a, b)


def test_ceros_numericos_distintos_son_equivalentes():
    # 2.8e-14 y 7.1e-15 difieren 4x en relativo pero ambos son "cero".
    assert equivalent("max diff=2.84e-14", "max diff=7.11e-15")


def test_cambio_de_texto_no_numerico_se_detecta():
    assert not equivalent("resultado: True", "resultado: False")


def test_columna_o_linea_nueva_se_detecta():
    assert not equivalent("a=1.0 b=2.0", "a=1.0 b=2.0 c=3.0")


def test_diferencia_relativa_justo_sobre_la_tolerancia_se_detecta():
    # 1e-4 relativo: por encima de RTOL=1e-5, debe considerarse cambio real.
    assert not equivalent("x=1.0000000", "x=1.0001000")


def test_diferencia_relativa_bajo_la_tolerancia_es_equivalente():
    # 1e-6 relativo: ruido, no cambio.
    assert equivalent("x=1.0000000", "x=1.0000010")
