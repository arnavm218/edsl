"""Compare simulated survey answers with real human answers, question by question.

ResultsFidelity takes two Results objects for the same questions: one from running a
survey with AI agents, and one holding human responses (for example from
Coop().get_human_survey_responses()). For each question it turns the answers on each
side into a distribution over the answer options and reports how far apart the two
distributions are.

Two distances are available, both between 0 (identical) and 1 (no overlap):

- "tvd", total variation distance: half the sum of the absolute differences in the
  share of each answer. It is the largest gap in the share of respondents that any
  set of answers can have.
- "jsd", Jensen-Shannon distance: the square root of the Jensen-Shannon divergence,
  computed with base-2 logarithms.

Results can be split by a subgroup field (an agent trait or a scenario key), and the
simulated side is always split by model. Survey weights can be given for either side.

Two optional extras help read the numbers:

- A bootstrap confidence interval for each distance. Respondents are resampled with
  replacement, separately on each side, and the distance recomputed; the interval is
  the matching percentiles of those distances. It shows how much the distance could
  move from sampling noise alone. The distance is biased upward for small samples
  (two samples from the same population rarely match exactly), and the interval does
  not correct for this, so it rarely reaches 0 even when the simulated and human
  answers come from the same population. An interval that excludes 0 is therefore
  not evidence of a difference; compare the distance with the split-half reference
  instead.
- A human split-half reference. The human respondents are randomly split into two
  halves and the distance between the halves computed, averaged over many splits. It
  estimates how different two groups of real people from the same population look,
  so a simulated distance near it is about as close as sampling noise allows. Each
  half has only half the respondents, so the reference is larger than the noise in
  a comparison that uses the full human sample: about 1.4 (the square root of 2)
  times larger when the simulated and human samples are the same size.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Hashable, Optional, Sequence

from ..exceptions import ResultsError

if TYPE_CHECKING:
    from ...scenarios import ScenarioList
    from ..result import Result
    from ..results import Results

# Question types whose answer is a single option, so answers form a distribution.
SUPPORTED_QUESTION_TYPES = (
    "multiple_choice",
    "yes_no",
    "linear_scale",
    "likert_five",
    "dropdown",
)

# One respondent's answer and survey weight.
Response = tuple[Hashable, float]
Distribution = dict[Hashable, float]


def distribution(responses: Sequence[Response]) -> Optional[Distribution]:
    """Return the weighted share of each answer, or None if the weights sum to zero.

    >>> distribution([("a", 1.0), ("b", 1.0), ("a", 2.0)])
    {'a': 0.75, 'b': 0.25}
    >>> distribution([]) is None
    True
    """
    total = sum(weight for _, weight in responses)
    if total <= 0:
        return None
    shares: dict[Hashable, float] = defaultdict(float)
    for answer, weight in responses:
        shares[answer] += weight / total
    return dict(shares)


def total_variation_distance(p: Distribution, q: Distribution) -> float:
    """Return half the sum of absolute differences between two distributions.

    >>> total_variation_distance({"a": 0.5, "b": 0.5}, {"a": 1.0})
    0.5
    >>> total_variation_distance({"a": 1.0}, {"b": 1.0})
    1.0
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def jensen_shannon_distance(p: Distribution, q: Distribution) -> float:
    """Return the Jensen-Shannon distance (base 2), between 0 and 1.

    >>> jensen_shannon_distance({"a": 1.0}, {"b": 1.0})
    1.0
    >>> jensen_shannon_distance({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5})
    0.0
    """
    divergence = 0.0
    for k in set(p) | set(q):
        pk, qk = p.get(k, 0.0), q.get(k, 0.0)
        mk = (pk + qk) / 2
        if pk > 0:
            divergence += 0.5 * pk * math.log2(pk / mk)
        if qk > 0:
            divergence += 0.5 * qk * math.log2(qk / mk)
    # Rounding can leave a tiny negative divergence for identical distributions.
    return math.sqrt(max(divergence, 0.0))


METRICS: dict[str, Callable[[Distribution, Distribution], float]] = {
    "tvd": total_variation_distance,
    "jsd": jensen_shannon_distance,
}


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Return the q-quantile, interpolating linearly between neighbours.

    >>> _quantile([0.0, 1.0, 2.0, 3.0], 0.5)
    1.5
    """
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


class ResultsFidelity:
    """Measures how closely simulated answers match human answers.

    >>> from edsl.results import Results
    >>> r = Results.example()
    >>> report = ResultsFidelity(simulated=r, human=r).compare()
    >>> [(row["question_name"], row["model"], row["distance"]) for row in report]
    [('how_feeling', 'test', 0.0), ('how_feeling_yesterday', 'test', 0.0)]
    """

    def __init__(self, simulated: "Results", human: "Results"):
        """Store the simulated (AI) results and the human results to compare."""
        self.simulated = simulated
        self.human = human

    def compare(
        self,
        questions: Optional[Sequence[str]] = None,
        metric: str = "tvd",
        by: Optional[str] = None,
        human_weight: Optional[str] = None,
        simulated_weight: Optional[str] = None,
        n_bootstrap: int = 0,
        confidence: float = 0.95,
        n_split_half: int = 0,
        seed: Optional[int] = None,
    ) -> "ScenarioList":
        """Return the distance between simulated and human answers for each question.

        Args:
            questions: Question names to compare. Defaults to every question in both
                surveys whose type has a single-option answer (multiple choice, yes/no,
                linear scale, Likert, dropdown).
            metric: "tvd" (total variation distance) or "jsd" (Jensen-Shannon
                distance).
            by: An agent trait or scenario key to compare within, such as "region".
                Prefix with "agent." or "scenario." to say where to look; otherwise
                agent traits are checked first.
            human_weight: An agent trait or scenario key holding each human
                respondent's survey weight. Unweighted if None.
            simulated_weight: The same for simulated respondents.
            n_bootstrap: Number of bootstrap resamples for a confidence interval. 0
                skips the interval.
            confidence: Coverage of the bootstrap interval, e.g. 0.95.
            n_split_half: Number of random splits of the human respondents used for
                the split-half reference. 0 skips it.
            seed: Seed for the bootstrap and split-half random draws.

        Returns:
            A ScenarioList with one row per question, model and subgroup, holding
            question_name, model, group, metric, distance, ci_lower, ci_upper,
            human_split_half, n_simulated and n_human. distance is None when either
            side has no answers; columns that were not requested are None. Answers
            of None (skipped or failed questions) are left out.

        >>> from edsl.results import Results
        >>> r = Results.example()
        >>> report = ResultsFidelity(r, r).compare(
        ...     questions=["how_feeling"], by="status", n_split_half=20, seed=0
        ... )
        >>> [(row["group"], row["distance"], row["n_human"]) for row in report]
        [('Joyful', 0.0, 2), ('Sad', 0.0, 2)]
        >>> [row["human_split_half"] for row in report]  # 2 different answers per group
        [1.0, 1.0]
        """
        if metric not in METRICS:
            raise ResultsError(
                f"Unknown metric {metric!r}; use one of {', '.join(METRICS)}."
            )
        if not 0 < confidence < 1:
            raise ResultsError("confidence must be between 0 and 1.")
        if n_bootstrap < 0 or n_split_half < 0:
            raise ResultsError("n_bootstrap and n_split_half cannot be negative.")
        distance = METRICS[metric]
        rng = random.Random(seed)

        rows: list[dict[str, Any]] = []
        for question in self._questions(questions):
            simulated = self._collect(
                self.simulated, question, by, simulated_weight, by_model=True
            )
            human = self._collect(self.human, question, by, human_weight)
            models = sorted({model for model, _ in simulated}, key=str)
            groups = sorted(
                {group for _, group in simulated} | {group for _, group in human},
                key=str,
            )
            # The human side does not depend on the model, so compute its
            # split-half reference once per group and reuse it.
            split_half: dict[Hashable, Optional[float]] = {}
            for model in models:
                for group in groups:
                    sim = simulated.get((model, group), [])
                    hum = human.get((None, group), [])
                    if n_split_half and group not in split_half:
                        split_half[group] = self._split_half(
                            hum, distance, n_split_half, rng
                        )
                    row = {
                        "question_name": question,
                        "model": model,
                        "group": group,
                        "metric": metric,
                        "distance": self._distance(sim, hum, distance),
                        "ci_lower": None,
                        "ci_upper": None,
                        "human_split_half": split_half.get(group),
                        "n_simulated": len(sim),
                        "n_human": len(hum),
                    }
                    if n_bootstrap and row["distance"] is not None:
                        row["ci_lower"], row["ci_upper"] = self._bootstrap(
                            sim, hum, distance, n_bootstrap, confidence, rng
                        )
                    rows.append(row)
        return self._scenario_list(rows)

    def distributions(
        self,
        question: str,
        by: Optional[str] = None,
        human_weight: Optional[str] = None,
        simulated_weight: Optional[str] = None,
    ) -> "ScenarioList":
        """Return the share of each answer on each side, to see where they differ.

        One row per model, subgroup and answer option, holding question_name, model,
        group, answer, simulated_share and human_share. Answer options that nobody
        chose have a share of 0. A share is None when that side has no answers.

        >>> from edsl.results import Results
        >>> r = Results.example()
        >>> rows = ResultsFidelity(r, r).distributions("how_feeling")
        >>> [(row["answer"], row["simulated_share"]) for row in rows]
        [('Good', 0.0), ('Great', 0.25), ('OK', 0.5), ('Terrible', 0.25)]
        """
        self._questions([question])
        simulated = self._collect(
            self.simulated, question, by, simulated_weight, by_model=True
        )
        human = self._collect(self.human, question, by, human_weight)
        options = self._options(question, simulated, human)
        models = sorted({model for model, _ in simulated}, key=str)
        groups = sorted(
            {group for _, group in simulated} | {group for _, group in human}, key=str
        )

        rows: list[dict[str, Any]] = []
        for model in models:
            for group in groups:
                sim = distribution(simulated.get((model, group), []))
                hum = distribution(human.get((None, group), []))
                for option in options:
                    rows.append(
                        {
                            "question_name": question,
                            "model": model,
                            "group": group,
                            "answer": option,
                            "simulated_share": (
                                None if sim is None else sim.get(option, 0.0)
                            ),
                            "human_share": (
                                None if hum is None else hum.get(option, 0.0)
                            ),
                        }
                    )
        return self._scenario_list(rows)

    def _questions(self, questions: Optional[Sequence[str]]) -> list[str]:
        """Return the questions to compare, checking that each one can be compared."""
        simulated_names = self.simulated.survey.question_names
        human_names = set(self.human.survey.question_names)
        if questions is None:
            return [
                name
                for name in simulated_names
                if name in human_names
                and self._question_type(self.simulated, name)
                in SUPPORTED_QUESTION_TYPES
            ]

        for name in questions:
            if name not in simulated_names or name not in human_names:
                raise ResultsError(
                    f"Question {name!r} must be in both the simulated and the human "
                    "survey."
                )
            question_type = self._question_type(self.simulated, name)
            if question_type not in SUPPORTED_QUESTION_TYPES:
                raise ResultsError(
                    f"Question {name!r} is a {question_type} question. ResultsFidelity "
                    "compares questions with one answer option per respondent ("
                    + ", ".join(SUPPORTED_QUESTION_TYPES)
                    + "). Numerical answers need binning into categories first, and "
                    "checkbox questions are not supported yet."
                )
        return list(questions)

    @staticmethod
    def _question_type(results: "Results", name: str) -> str:
        return results.survey.get(name).question_type

    @staticmethod
    def _collect(
        results: "Results",
        question: str,
        by: Optional[str],
        weight: Optional[str],
        by_model: bool = False,
    ) -> dict[tuple[Optional[str], Hashable], list[Response]]:
        """Group each respondent's answer and weight by (model, subgroup).

        The model is None unless by_model is True, and the subgroup is None unless by
        is given. Answers of None are left out.
        """
        groups: dict[tuple[Optional[str], Hashable], list[Response]] = defaultdict(list)
        for result in results:
            answer = result.answer.get(question)
            if answer is None:
                continue
            if not isinstance(answer, Hashable):
                raise ResultsError(
                    f"Question {question!r} has an answer of type "
                    f"{type(answer).__name__}; ResultsFidelity needs one option per "
                    "answer."
                )
            model = result.model.model if by_model else None
            group = ResultsFidelity._field(result, by) if by else None
            groups[(model, group)].append(
                (answer, ResultsFidelity._weight(result, weight))
            )
        return groups

    @staticmethod
    def _field(result: "Result", name: str) -> Any:
        """Return an agent trait or scenario value from one result."""
        sources = {
            "agent": result.agent.traits,
            "scenario": result.scenario,
        }
        prefix, _, key = name.partition(".")
        if key and prefix in sources:
            if key in sources[prefix]:
                return sources[prefix][key]
        else:
            for values in sources.values():
                if name in values:
                    return values[name]
        raise ResultsError(
            f"{name!r} is not an agent trait or scenario key of every respondent."
        )

    @staticmethod
    def _weight(result: "Result", name: Optional[str]) -> float:
        """Return a respondent's survey weight, 1.0 when no weight field is given."""
        if name is None:
            return 1.0
        value = ResultsFidelity._field(result, name)
        try:
            weight = float(value)
        except (TypeError, ValueError):
            weight = math.nan
        if not math.isfinite(weight) or weight < 0:
            raise ResultsError(
                f"Weight {name!r} must be a non-negative number, not {value!r}."
            )
        return weight

    @staticmethod
    def _distance(
        simulated: Sequence[Response],
        human: Sequence[Response],
        distance: Callable[[Distribution, Distribution], float],
    ) -> Optional[float]:
        p, q = distribution(simulated), distribution(human)
        if p is None or q is None:
            return None
        return distance(p, q)

    @staticmethod
    def _bootstrap(
        simulated: Sequence[Response],
        human: Sequence[Response],
        distance: Callable[[Distribution, Distribution], float],
        n_bootstrap: int,
        confidence: float,
        rng: random.Random,
    ) -> tuple[Optional[float], Optional[float]]:
        """Return a percentile bootstrap interval for the distance.

        Each resample draws respondents with replacement, separately from each side
        and keeping each side's size, then recomputes the distance.
        """
        values = []
        for _ in range(n_bootstrap):
            value = ResultsFidelity._distance(
                rng.choices(simulated, k=len(simulated)),
                rng.choices(human, k=len(human)),
                distance,
            )
            # A resample can pick only zero-weight respondents; skip it.
            if value is not None:
                values.append(value)
        if not values:
            return None, None
        values.sort()
        tail = (1 - confidence) / 2
        return _quantile(values, tail), _quantile(values, 1 - tail)

    @staticmethod
    def _split_half(
        human: Sequence[Response],
        distance: Callable[[Distribution, Distribution], float],
        n_splits: int,
        rng: random.Random,
    ) -> Optional[float]:
        """Return the mean distance between two random halves of the human answers."""
        if len(human) < 2:
            return None
        values = []
        shuffled = list(human)
        for _ in range(n_splits):
            rng.shuffle(shuffled)
            middle = len(shuffled) // 2
            value = ResultsFidelity._distance(
                shuffled[:middle], shuffled[middle:], distance
            )
            if value is not None:
                values.append(value)
        return sum(values) / len(values) if values else None

    def _options(
        self,
        question: str,
        simulated: dict[Any, list[Response]],
        human: dict[Any, list[Response]],
    ) -> list[Hashable]:
        """Return the question's options, followed by any other answers given."""
        options: list[Hashable] = []
        for results in (self.simulated, self.human):
            for option in (
                getattr(results.survey.get(question), "question_options", None) or []
            ):
                if option not in options:
                    options.append(option)
        given = {
            answer
            for responses in (*simulated.values(), *human.values())
            for answer, _ in responses
        }
        return options + sorted(given - set(options), key=str)

    @staticmethod
    def _scenario_list(rows: list[dict[str, Any]]) -> "ScenarioList":
        from ...scenarios import Scenario, ScenarioList

        return ScenarioList([Scenario(row) for row in rows])
