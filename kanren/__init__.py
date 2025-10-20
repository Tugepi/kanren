# flake8: noqa
"""kanren is a Python library for logic and relational programming."""
from importlib.metadata import version

from unification import Var, isvar, reify, unifiable, unify, var, variables, vars

from .core import conde, eq, lall, lany, run
from .core import run_foreign, pred_to_results_filter, run_foreign2
from .facts import Relation, fact, facts
from .goals import (
    appendo,
    conso,
    heado,
    itero,
    membero,
    nullo,
    permuteo,
    permuteq,
    rembero,
    tailo,
)
from .term import arguments, operator, term, unifiable_with_term

#for the ffi
from .constraints import (
    ConstrainedState,
    ConstraintStore,
    PredicateStore,
    DisequalityStore,
    TypeStore,
    IsinstanceStore,
    neq,
    typeo,
    isinstanceo,
    foreigno,
    FunctionConstraintStore,
    all_function_constraints_resolved,
)

__version__ = version("miniKanren")
