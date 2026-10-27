"""Tes parser dump SQL data apotek (tanpa perlu data mentah)."""

import pytest

from retailfc.id_pharmacy import parse_dump, parse_tuple

DUMP = (
    "-- komentar\r\n"
    "DELETE FROM `ms_product`;\r\n"
    "INSERT INTO `ms_product` (`KD_OBAT`, `NAMA`, `HJ_RP`) VALUES\r\n"
    "\t('AI-0001', 'Obat, 500mg ', 0),\r\n"
    "\t('AI-0002', 'Kutip \\'a\\' dan ''b''', 184000.5),\r\n"
    "\t('AI-0003', NULL, -3);\r\n"
    "INSERT INTO `ms_sales` (`NO_RESEP`, `TGL`) VALUES\r\n"
    "\t('R-1', '2015-01-02');\r\n"
)


def test_parse_tuple_values():
    vals, rest = parse_tuple("\t('a, b)', 1, 2.5, NULL, -4),")
    assert vals == ["a, b)", 1, 2.5, None, -4]
    assert rest == ","


def test_parse_dump(tmp_path):
    p = tmp_path / "d.sql"
    p.write_bytes(DUMP.encode("utf-8"))
    t = parse_dump(p)
    prod = t["ms_product"]
    assert list(prod.columns) == ["KD_OBAT", "NAMA", "HJ_RP"]
    assert prod["NAMA"].tolist()[:2] == ["Obat, 500mg ", "Kutip 'a' dan 'b'"]
    assert prod["NAMA"].isna().tolist() == [False, False, True]
    assert prod["HJ_RP"].tolist() == [0, 184000.5, -3]
    assert str(t["ms_sales"]["TGL"].iloc[0].date()) == "2015-01-02"


def test_parse_dump_rejects_wrong_width(tmp_path):
    p = tmp_path / "bad.sql"
    p.write_text("INSERT INTO `x` (`a`, `b`) VALUES\n\t('1');\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1 nilai"):
        parse_dump(p)
