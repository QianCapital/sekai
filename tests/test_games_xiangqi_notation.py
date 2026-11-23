"""Xiangqi WXF and Chinese move notation: openings, every piece and operator, tandem pieces."""

from __future__ import annotations

import pytest

from sekai.envs.games.xiangqi.engine import START_FEN, Position
from sekai.envs.games.xiangqi.errors import IllegalMove
from sekai.envs.games.xiangqi.notation import describe_move, file_number, to_chinese, to_wxf


def after(*moves: str, fen: str = START_FEN) -> Position:
    position = Position(fen)
    for uci in moves:
        position.push_uci(uci)
    return position


def notate(position: Position, uci: str) -> tuple[str, str]:
    return to_wxf(position, uci), to_chinese(position, uci)


def test_file_numbers_count_from_each_side_right() -> None:
    assert [file_number(f, 1) for f in range(9)] == [9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert [file_number(f, -1) for f in range(9)] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize(
    ("uci", "wxf", "chinese"),
    [
        ("h2e2", "C2=5", "炮二平五"),  # central cannon
        ("b2e2", "C8=5", "炮八平五"),
        ("h0g2", "H2+3", "馬二進三"),
        ("b0c2", "H8+7", "馬八進七"),
        ("c3c4", "P7+1", "兵七進一"),  # 仙人指路 (pawn opening)
        ("g3g4", "P3+1", "兵三進一"),
        ("g0e2", "E3+5", "相三進五"),  # elephant opening
        ("c0e2", "E7+5", "相七進五"),
        ("f0e1", "A4+5", "仕四進五"),
        ("d0e1", "A6+5", "仕六進五"),
        ("e0e1", "K5+1", "帥五進一"),
        ("i0i1", "R1+1", "車一進一"),
        ("a0a2", "R9+2", "車九進二"),
        ("b2b9", "C8+7", "炮八進七"),  # cannon takes horse
        ("h2h9", "C2+7", "炮二進七"),
        ("b2b1", "C8-1", "炮八退一"),
    ],
)
def test_red_moves_from_the_start(uci: str, wxf: str, chinese: str) -> None:
    assert notate(Position(), uci) == (wxf, chinese)


@pytest.mark.parametrize(
    ("uci", "wxf", "chinese"),
    [
        ("h9g7", "H8+7", "馬8進7"),  # screen-horse defence
        ("b9c7", "H2+3", "馬2進3"),
        ("h7e7", "C8=5", "砲8平5"),  # same-direction cannons
        ("b7e7", "C2=5", "砲2平5"),
        ("c6c5", "P3+1", "卒3進1"),
        ("g6g5", "P7+1", "卒7進1"),
        ("a9a8", "R1+1", "車1進1"),
        ("d9e8", "A4+5", "士4進5"),
        ("c9e7", "E3+5", "象3進5"),
        ("e9e8", "K5+1", "將5進1"),
        ("b7b8", "C2-1", "砲2退1"),
    ],
)
def test_black_replies(uci: str, wxf: str, chinese: str) -> None:
    assert notate(after("h2e2"), uci) == (wxf, chinese)


def test_classic_opening_sequence() -> None:
    # 1. 炮二平五 馬8進7 2. 馬二進三 車9平8 3. 車一平二 馬2進3 4. 兵七進一 卒7進1
    line = [
        ("h2e2", "C2=5", "炮二平五"),
        ("h9g7", "H8+7", "馬8進7"),
        ("h0g2", "H2+3", "馬二進三"),
        ("i9h9", "R9=8", "車9平8"),
        ("i0h0", "R1=2", "車一平二"),
        ("b9c7", "H2+3", "馬2進3"),
        ("c3c4", "P7+1", "兵七進一"),
        ("g6g5", "P7+1", "卒7進1"),
        ("h0h6", "R2+6", "車二進六"),
        ("h7i7", "C8=9", "砲8平9"),
        ("h6g6", "R2=3", "車二平三"),
        ("i7i8", "C9-1", "砲9退1"),
        ("f0e1", "A4+5", "仕四進五"),
        ("e9e8", "K5+1", "將5進1"),
        ("e0f0", "K5=4", "帥五平四"),
        ("e8e9", "K5-1", "將5退1"),
        ("b2b1", "C8-1", "炮八退一"),
    ]
    position = Position()
    for uci, wxf, chinese in line:
        assert notate(position, uci) == (wxf, chinese), uci
        position.push_uci(uci)


def test_horse_and_diagonal_retreats() -> None:
    position = after("h0g2", "h9g7")
    assert notate(position, "g2h0") == ("H3-2", "馬三退二")
    position = after("g0e2", "c9e7")
    assert notate(position, "e2g0") == ("E5-3", "相五退三")
    position = after("f0e1", "f9e8")
    assert notate(position, "e1f0") == ("A5-4", "仕五退四")
    position.push_uci("b0c2")
    assert notate(position, "e8f9") == ("A5-6", "士5退6")


def test_tandem_cannons() -> None:
    # Red cannons on h4 (front) and h2 (rear).
    position = Position("rnbakabnr/9/1c5c1/p1p1p1p1p/9/7C1/P1P1P1P1P/7C1/9/RNBAKABNR w - - 0 1")
    assert notate(position, "h4e4") == ("+C=5", "前炮平五")
    assert notate(position, "h2e2") == ("-C=5", "後炮平五")
    assert notate(position, "h4h5") == ("+C+1", "前炮進一")
    assert notate(position, "h2h1") == ("-C-1", "後炮退一")
    assert notate(position, "b0c2") == ("H8+7", "馬八進七")  # other pieces unaffected


def test_tandem_black_chariots() -> None:
    # Black chariots on a8 (front: closer to Red) and a9 (rear).
    position = Position("r1bakabn1/r8/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1")
    assert notate(position, "a8b8") == ("+R=2", "前車平2")
    assert notate(position, "a9b9") == ("-R=2", "後車平2")
    assert notate(position, "a8a7") == ("+R+1", "前車進1")


def test_tandem_horses() -> None:
    position = Position("4k4/9/9/9/9/2N6/9/2N6/9/3K5 w - - 0 1")
    assert notate(position, "c4d2") == ("+H-6", "前馬退六")
    assert notate(position, "c2e3") == ("-H+5", "後馬進五")


def test_three_soldiers_on_one_file() -> None:
    position = Position("5k3/9/4P4/4P4/4P4/9/9/9/9/3K5 w - - 0 1")
    assert notate(position, "e7d7") == ("+P=6", "前兵平六")
    assert notate(position, "e7e8") == ("+P+1", "前兵進一")
    assert notate(position, "e6d6") == ("=P=6", "中兵平六")
    assert notate(position, "e5f5") == ("-P=4", "後兵平四")


def test_four_soldiers_on_one_file() -> None:
    position = Position("3k5/4P4/4P4/4P4/4P4/9/9/9/9/5K3 w - - 0 1")
    assert notate(position, "e8f8") == ("1P=4", "一兵平四")
    assert notate(position, "e7d7") == ("2P=6", "二兵平六")
    assert notate(position, "e6f6") == ("3P=4", "三兵平四")
    assert notate(position, "e5d5") == ("4P=6", "四兵平六")


def test_soldier_tandems_on_two_files() -> None:
    position = Position("5k3/9/2P6/2P1P4/4P4/9/9/9/9/3K5 w - - 0 1")
    assert notate(position, "c7b7") == ("+7=8", "前七平八")
    assert notate(position, "c6b6") == ("-7=8", "後七平八")
    assert notate(position, "e6e7") == ("+5+1", "前五進一")
    assert notate(position, "e5d5") == ("-5=6", "後五平六")
    black = Position("3k5/9/9/9/9/4p4/2p1p4/2p6/9/5K3 b - - 0 1")
    assert notate(black, "c2c1") == ("+3+1", "前3進1")
    assert notate(black, "c3b3") == ("-3=2", "後3平2")
    assert notate(black, "e4d4") == ("-5=4", "後5平4")


def test_advisors_and_elephants_on_one_file_use_the_operator() -> None:
    advisors = Position("5k3/9/9/9/R8/9/9/3A5/9/3AK4 w - - 0 1")
    assert notate(advisors, "d0e1") == ("A6+5", "仕六進五")
    assert notate(advisors, "d2e1") == ("A6-5", "仕六退五")
    elephants = Position("5k3/9/9/9/R8/2B6/9/9/9/2B1K4 w - - 0 1")
    assert notate(elephants, "c0e2") == ("E7+5", "相七進五")
    assert notate(elephants, "c4e2") == ("E7-5", "相七退五")
    assert describe_move(elephants, "c4e2").tandem == 1


def test_notation_requires_a_legal_move() -> None:
    with pytest.raises(IllegalMove):
        to_wxf(Position(), "h2h8")  # a cannon cannot move quietly past its screen
    with pytest.raises(IllegalMove):
        to_chinese(Position(), "h9g7")  # Black's move, Red to play
    with pytest.raises(ValueError):
        to_wxf(Position(), "xx")
