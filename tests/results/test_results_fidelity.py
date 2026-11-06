"""ResultsFidelity: comparing simulated answer distributions with human ones."""

import copy
import json
import math

import pytest

from edsl import (
    QuestionCheckBox,
    QuestionFreeText,
    QuestionLinearScale,
    QuestionMultipleChoice,
    QuestionNumerical,
    Survey,
)
from edsl.agents import Agent
from edsl.language_models import Model
from edsl.results import Result, Results
from edsl.results.exceptions import ResultsError
from edsl.results.extras import ResultsFidelity
from edsl.results.extras.results_fidelity import (
    distribution,
    jensen_shannon_distance,
    total_variation_distance,
)
from edsl.scenarios import Scenario


def make_survey():
    return Survey(
        [
            QuestionMultipleChoice(
                question_name="color",
                question_text="Favorite color?",
                question_options=["red", "blue", "green"],
            ),
            QuestionLinearScale(
                question_name="rating",
                question_text="Rate it.",
                question_options=[1, 2, 3, 4, 5],
            ),
            QuestionFreeText(question_name="why", question_text="Why?"),
            QuestionCheckBox(
                question_name="pets",
                question_text="Which pets?",
                question_options=["cat", "dog"],
            ),
            QuestionNumerical(question_name="age", question_text="Age?"),
        ]
    )


def model_named(name):
    model = copy.deepcopy(Model("test"))
    model.model = name
    return model


def make_results(colors, traits=None, scenarios=None, model="test", survey=None):
    """Build Results with one respondent per entry in colors.

    Other questions get fixed answers so they never affect the color comparison.
    """
    survey = survey or make_survey()
    llm = model_named(model)
    attributes = survey.question_to_attributes()
    data = []
    for i, color in enumerate(colors):
        data.append(
            Result(
                agent=Agent(traits=(traits[i] if traits else {})),
                scenario=Scenario(scenarios[i] if scenarios else {}),
                model=llm,
                iteration=0,
                answer={
                    "color": color,
                    "rating": 3,
                    "why": "because",
                    "pets": ["cat"],
                    "age": 30,
                },
                survey=survey,
                question_to_attributes=attributes,
            )
        )
    return Results(survey=survey, data=data)


def only(report):
    rows = list(report)
    assert len(rows) == 1
    return rows[0]


# --- distance functions ---------------------------------------------------------


def test_distribution_is_weighted_share():
    assert distribution([("a", 1), ("b", 3)]) == {"a": 0.25, "b": 0.75}
    assert distribution([("a", 0)]) is None


def test_tvd_and_jsd_bounds():
    same = {"a": 0.3, "b": 0.7}
    assert total_variation_distance(same, same) == 0
    assert jensen_shannon_distance(same, same) == 0
    assert total_variation_distance({"a": 1.0}, {"b": 1.0}) == 1
    assert jensen_shannon_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1)


def test_jsd_matches_formula():
    p, q = {"a": 0.5, "b": 0.5}, {"a": 0.75, "b": 0.25}
    m = {"a": 0.625, "b": 0.375}
    divergence = 0.5 * sum(p[k] * math.log2(p[k] / m[k]) for k in p) + 0.5 * sum(
        q[k] * math.log2(q[k] / m[k]) for k in q
    )
    assert jensen_shannon_distance(p, q) == pytest.approx(math.sqrt(divergence))


# --- compare ----------------------------------------------------------------


def test_identical_answers_have_zero_distance():
    results = make_results(["red", "blue", "blue"])
    report = ResultsFidelity(results, results).compare(questions=["color"])
    assert only(report)["distance"] == 0


def test_known_tvd():
    # simulated: red 1/2, blue 1/2; human: red 3/4, blue 1/4 -> TVD 1/4
    simulated = make_results(["red", "red", "blue", "blue"])
    human = make_results(["red", "red", "red", "blue"])
    row = only(ResultsFidelity(simulated, human).compare(questions=["color"]))
    assert row["distance"] == pytest.approx(0.25)
    assert row["metric"] == "tvd"
    assert (row["n_simulated"], row["n_human"]) == (4, 4)
    assert row["model"] == "test"
    assert row["group"] is None


def test_jsd_metric():
    simulated = make_results(["red", "red", "blue", "blue"])
    human = make_results(["red", "red", "red", "blue"])
    row = only(
        ResultsFidelity(simulated, human).compare(questions=["color"], metric="jsd")
    )
    expected = jensen_shannon_distance(
        {"red": 0.5, "blue": 0.5}, {"red": 0.75, "blue": 0.25}
    )
    assert row["distance"] == pytest.approx(expected)
    assert row["metric"] == "jsd"


def test_default_questions_skip_unsupported_types():
    results = make_results(["red"])
    report = ResultsFidelity(results, results).compare()
    assert [row["question_name"] for row in report] == ["color", "rating"]


@pytest.mark.parametrize("question", ["pets", "age", "why"])
def test_unsupported_question_raises(question):
    results = make_results(["red"])
    with pytest.raises(ResultsError, match="not supported yet|binning"):
        ResultsFidelity(results, results).compare(questions=[question])


def test_question_missing_from_human_survey_raises():
    simulated = make_results(["red"])
    human_survey = Survey(
        [
            QuestionMultipleChoice(
                question_name="other", question_text="?", question_options=["x"]
            )
        ]
    )
    human = Results(survey=human_survey, data=[])
    with pytest.raises(ResultsError, match="both"):
        ResultsFidelity(simulated, human).compare(questions=["color"])


def test_bad_arguments_raise():
    results = make_results(["red"])
    fidelity = ResultsFidelity(results, results)
    with pytest.raises(ResultsError, match="metric"):
        fidelity.compare(metric="kl")
    with pytest.raises(ResultsError, match="confidence"):
        fidelity.compare(confidence=1.5)
    with pytest.raises(ResultsError, match="negative"):
        fidelity.compare(n_bootstrap=-1)


def test_none_answers_are_left_out():
    simulated = make_results(["red", None, "blue"])
    human = make_results(["red", "blue", None, None])
    row = only(ResultsFidelity(simulated, human).compare(questions=["color"]))
    assert row["distance"] == 0
    assert (row["n_simulated"], row["n_human"]) == (2, 2)


def test_one_row_per_model():
    human = make_results(["red", "blue"])
    simulated = make_results(["red", "blue"], model="model-a") + make_results(
        ["red", "red"], model="model-b"
    )
    report = ResultsFidelity(simulated, human).compare(questions=["color"])
    assert [(row["model"], row["distance"]) for row in report] == [
        ("model-a", 0),
        ("model-b", pytest.approx(0.5)),
    ]


# --- subgroups --------------------------------------------------------------


def test_by_agent_trait():
    traits = [{"region": "north"}, {"region": "north"}, {"region": "south"}]
    simulated = make_results(["red", "red", "blue"], traits=traits)
    human = make_results(["red", "blue", "blue"], traits=traits)
    report = ResultsFidelity(simulated, human).compare(questions=["color"], by="region")
    # north: simulated all red, human half red -> 0.5; south: both blue -> 0
    assert [(row["group"], row["distance"]) for row in report] == [
        ("north", pytest.approx(0.5)),
        ("south", 0),
    ]


def test_by_scenario_key_with_prefix():
    scenarios = [{"wave": 1}, {"wave": 2}]
    simulated = make_results(["red", "blue"], scenarios=scenarios)
    human = make_results(["red", "red"], scenarios=scenarios)
    report = ResultsFidelity(simulated, human).compare(
        questions=["color"], by="scenario.wave"
    )
    assert [(row["group"], row["distance"]) for row in report] == [(1, 0), (2, 1)]


def test_group_on_one_side_only_has_no_distance():
    simulated = make_results(["red"], traits=[{"region": "north"}])
    human = make_results(
        ["red", "red"], traits=[{"region": "north"}, {"region": "west"}]
    )
    report = ResultsFidelity(simulated, human).compare(questions=["color"], by="region")
    rows = {row["group"]: row for row in report}
    assert rows["north"]["distance"] == 0
    assert rows["west"]["distance"] is None
    assert (rows["west"]["n_simulated"], rows["west"]["n_human"]) == (0, 1)


def test_missing_by_field_raises():
    results = make_results(["red"])
    with pytest.raises(ResultsError, match="region"):
        ResultsFidelity(results, results).compare(questions=["color"], by="region")


# --- weights ----------------------------------------------------------------


def test_human_weights():
    simulated = make_results(["red", "blue"])
    # Weighted human shares: red 3/4, blue 1/4 -> TVD 1/4 from simulated 1/2, 1/2.
    human = make_results(["red", "blue"], traits=[{"w": 3}, {"w": 1}])
    fidelity = ResultsFidelity(simulated, human)
    assert only(fidelity.compare(questions=["color"]))["distance"] == 0
    weighted = only(fidelity.compare(questions=["color"], human_weight="w"))
    assert weighted["distance"] == pytest.approx(0.25)
    # n counts respondents, not weight.
    assert weighted["n_human"] == 2


def test_simulated_weights():
    simulated = make_results(["red", "blue"], traits=[{"w": 1}, {"w": 3}])
    human = make_results(["red", "blue", "blue", "blue"])
    row = only(
        ResultsFidelity(simulated, human).compare(
            questions=["color"], simulated_weight="w"
        )
    )
    assert row["distance"] == pytest.approx(0)


@pytest.mark.parametrize("bad", [-1, "heavy", None, float("nan")])
def test_bad_weight_raises(bad):
    human = make_results(["red"], traits=[{"w": bad}])
    with pytest.raises(ResultsError, match="non-negative"):
        ResultsFidelity(human, human).compare(questions=["color"], human_weight="w")


# --- bootstrap and split-half -----------------------------------------------


def test_bootstrap_interval():
    simulated = make_results(["red"] * 10 + ["blue"] * 10)
    human = make_results(["red"] * 15 + ["blue"] * 5)
    fidelity = ResultsFidelity(simulated, human)
    row = only(fidelity.compare(questions=["color"], n_bootstrap=200, seed=1))
    assert 0 <= row["ci_lower"] <= row["ci_upper"] <= 1
    assert row["ci_lower"] < row["distance"] < row["ci_upper"]
    # Same seed, same interval.
    again = only(fidelity.compare(questions=["color"], n_bootstrap=200, seed=1))
    assert (again["ci_lower"], again["ci_upper"]) == (row["ci_lower"], row["ci_upper"])


def test_bootstrap_of_constant_answers_is_zero_width():
    results = make_results(["red"] * 5)
    row = only(
        ResultsFidelity(results, results).compare(
            questions=["color"], n_bootstrap=50, seed=0
        )
    )
    assert (row["ci_lower"], row["ci_upper"]) == (0, 0)


def test_extras_are_off_by_default():
    results = make_results(["red", "blue"])
    row = only(ResultsFidelity(results, results).compare(questions=["color"]))
    assert row["ci_lower"] is None and row["ci_upper"] is None
    assert row["human_split_half"] is None


def test_split_half_reference():
    simulated = make_results(["red"])
    # Identical human answers: any two halves match.
    same = make_results(["blue"] * 6)
    row = only(
        ResultsFidelity(simulated, same).compare(
            questions=["color"], n_split_half=10, seed=0
        )
    )
    assert row["human_split_half"] == 0
    # Two respondents with different answers: the halves never overlap.
    split = make_results(["red", "blue"])
    row = only(
        ResultsFidelity(simulated, split).compare(
            questions=["color"], n_split_half=10, seed=0
        )
    )
    assert row["human_split_half"] == 1
    # One respondent cannot be split.
    one = make_results(["red"])
    row = only(
        ResultsFidelity(simulated, one).compare(
            questions=["color"], n_split_half=10, seed=0
        )
    )
    assert row["human_split_half"] is None


def test_split_half_shrinks_with_more_respondents():
    simulated = make_results(["red"])
    small = make_results(["red", "blue"] * 5)
    large = make_results(["red", "blue"] * 200)
    kwargs = dict(questions=["color"], n_split_half=50, seed=0)
    small_ref = only(ResultsFidelity(simulated, small).compare(**kwargs))
    large_ref = only(ResultsFidelity(simulated, large).compare(**kwargs))
    assert large_ref["human_split_half"] < small_ref["human_split_half"]


# --- distributions ----------------------------------------------------------


def test_distributions():
    simulated = make_results(["red", "red", "blue", "blue"])
    human = make_results(["red", "red", "red", "blue"])
    rows = ResultsFidelity(simulated, human).distributions("color")
    shares = {
        row["answer"]: (row["simulated_share"], row["human_share"]) for row in rows
    }
    assert shares == {
        "red": (0.5, 0.75),
        "blue": (0.5, 0.25),
        "green": (0, 0),
    }


# --- with human responses loaded the way Coop returns them ------------------


def test_human_results_from_stored_responses():
    survey = make_survey()
    rows = [
        {
            "response_uuid": f"r{i}",
            "response_json_string": json.dumps({"color": {"answer": color}}),
            "agent_traits_json_string": json.dumps({"region": region}),
        }
        for i, (color, region) in enumerate(
            [("red", "north"), ("blue", "north"), ("blue", "south")]
        )
    ]
    human = Results.from_human_responses(survey, rows)
    simulated = make_results(
        ["red", "red", "blue"],
        traits=[{"region": "north"}, {"region": "north"}, {"region": "south"}],
        survey=survey,
    )
    report = ResultsFidelity(simulated, human).compare(questions=["color"], by="region")
    assert [(row["group"], row["distance"]) for row in report] == [
        ("north", pytest.approx(0.5)),
        ("south", 0),
    ]
