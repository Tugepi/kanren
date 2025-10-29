import weakref
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Mapping
from typing import Optional

from cons.core import ConsPair
from toolz import groupby
from unification import Var, reify, unify, var
from unification.core import _reify, isground
from unification.utils import transitive_get as walk

from .util import FlexibleSet


class ConstraintStore(ABC):
    """A class that enforces constraints between logic variables in a miniKanren state.

    Attributes
    ----------
    lvar_constraints: MutableMapping
        A mapping of logic variables to sets of objects that define their
        constraints (e.g. a set of items with which the logic variable cannot
        be unified).  The mapping's values are entirely determined by the
        ConstraintStore implementation.

    """

    __slots__ = ("lvar_constraints",)
    op_str: Optional[str] = None

    def __init__(self, lvar_constraints=None):
        # self.lvar_constraints = weakref.WeakKeyDictionary(lvar_constraints)
        self.lvar_constraints = lvar_constraints or dict()

    @abstractmethod
    def pre_unify_check(self, lvar_map, lvar=None, value=None):
        """Check a key-value pair before they're added to a ConstrainedState."""
        raise NotImplementedError()

    @abstractmethod
    def post_unify_check(self, lvar_map, lvar=None, value=None, old_state=None):
        """Check a key-value pair after they're added to a ConstrainedState.

        XXX: This method may alter the internal constraints, so make a copy!
        """
        raise NotImplementedError()

    def add(self, lvar, lvar_constraint, **kwargs):
        """Add a new constraint."""
        if lvar not in self.lvar_constraints:
            self.lvar_constraints[lvar] = FlexibleSet([lvar_constraint])
        else:
            self.lvar_constraints[lvar].add(lvar_constraint)

    def constraints_str(self, lvar):
        """Print the constraints on a logic variable."""
        if lvar in self.lvar_constraints:
            return f"{self.op_str} {self.lvar_constraints[lvar]}"
        else:
            return ""

    def copy(self):
        return type(self)(
            lvar_constraints={k: v.copy() for k, v in self.lvar_constraints.items()},
        )

    def __contains__(self, lvar):
        return lvar in self.lvar_constraints

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.op_str == other.op_str
            and self.lvar_constraints == other.lvar_constraints
        )

    def __repr__(self):
        return f"ConstraintStore({self.op_str}: {self.lvar_constraints})"
    



class ConstrainedState(UserDict):
    """A miniKanren state that holds unifications of logic variables and upholds constraints on logic variables."""  # noqa: E501

    __slots__ = ("constraints",)

    def __init__(self, *s, constraints=None):
        super().__init__(*s)
        self.constraints = dict(constraints or [])

    def pre_unify_checks(self, lvar, value):
        """Check the constraints before unification."""
        return all(
            cstore.pre_unify_check(self.data, lvar, value)
            for cstore in self.constraints.values()
        )

    def post_unify_checks(self, lvar_map, lvar, value):
        """Check constraints and return an updated state and constraints.

        Returns
        -------
        A new `ConstrainedState` and `False`.

        """
        S = self.copy(data=lvar_map)
        if any(
            not cstore.post_unify_check(lvar_map, lvar, value, old_state=S)
            for cstore in S.constraints.values()
        ):
            return False

        return S

    def copy(self, data=None):
        if data is None:
            data = self.data.copy()
        return type(self)(
            data, constraints={k: v.copy() for k, v in self.constraints.items()}
        )

    def __eq__(self, other):
        if isinstance(other, ConstrainedState):
            return self.data == other.data and self.constraints == other.constraints

        if isinstance(other, Mapping) and not self.constraints:
            return self.data == other

        return False

    def __repr__(self):
        return f"ConstrainedState({repr(self.data)}, {self.constraints})"


def unify_ConstrainedState(u, v, S):
    if S.pre_unify_checks(u, v):
        s = unify(u, v, S.data)
        if s is not False:
            S = S.post_unify_checks(s, u, v)
            if S is not False:
                return S

    return False


unify.add((object, object, ConstrainedState), unify_ConstrainedState)


class ConstrainedVar(Var):
    """A logic variable that tracks its own constraints.

    Currently, this is only for display/reification purposes.

    """

    __slots__ = ("S", "var")

    def __init__(self, var, S):
        self.S = weakref.ref(S)
        self.token = var.token
        self.var = weakref.ref(var)

    def __repr__(self):
        S = self.S()
        var = self.var()
        res = super().__repr__()
        if S is not None and var is not None:
            u_constraints = ",".join(
                [c.constraints_str(var) for c in S.constraints.values()]
            )
            return f"{res}: {{{u_constraints}}}"
        else:
            return res

    def __eq__(self, other):
        if type(other) == type(self):
            return self.S == other.S and self.token == other.token
        elif type(other) == Var:
            # NOTE: A more valid comparison is same token and no constraints.
            return self.token == other.token
        return NotImplemented

    def __hash__(self):
        return hash((Var, self.token))


def _reify_ConstrainedState(u, S):
    u_res = walk(u, S.data)

    if u_res is u:
        yield ConstrainedVar(u_res, S)
    else:
        yield _reify(u_res, S)


_reify.add((Var, ConstrainedState), _reify_ConstrainedState)


class DisequalityStore(ConstraintStore):
    """A disequality constraint (i.e. two things do not unify)."""

    op_str = "neq"

    def __init__(self, lvar_constraints=None):
        super().__init__(lvar_constraints)

    def post_unify_check(self, lvar_map, lvar=None, value=None, old_state=None):

        for lv_key, constraints in list(self.lvar_constraints.items()):
            lv = reify(lv_key, lvar_map)
            constraints_rf = reify(tuple(constraints), lvar_map)

            for cs in constraints_rf:
                s = unify(lv, cs, {})

                if s is not False and not s:
                    # They already unify, but with no unground logic variables,
                    # so we have an immediate violation of the constraint.
                    return False
                elif s is False:
                    # They don't unify and have no unground logic variables, so
                    # the constraint is immediately satisfied and there's no
                    # reason to continue checking this constraint.
                    constraints.discard(cs)
                else:
                    # They unify when/if the unifications in `s` are made, so
                    # let's add these as new constraints.
                    for k, v in s.items():
                        self.add(k, v)

            if len(constraints) == 0:
                # This logic variable has no more unground constraints, so
                # remove it.
                del self.lvar_constraints[lv_key]

        return True

    def pre_unify_check(self, lvar_map, lvar=None, value=None):
        return True


def neq(u, v):
    """Construct a disequality goal."""

    def neq_goal(S):
        nonlocal u, v

        u_rf, v_rf = reify((u, v), S)

        # Get the unground logic variables that would unify the two objects;
        # these are all the logic variables that we can't let unify.
        s_uv = unify(u_rf, v_rf, {})

        if s_uv is False:
            # They don't unify and have no unground logic variables, so the
            # constraint is immediately satisfied.
            yield S
            return
        elif not s_uv:
            # They already unify, but with no unground logic variables, so we
            # have an immediate violation of the constraint.
            return

        if not isinstance(S, ConstrainedState):
            S = ConstrainedState(S)

        cs = S.constraints.setdefault(DisequalityStore, DisequalityStore())

        for lvar, obj in s_uv.items():
            cs.add(lvar, obj)

        # We need to check the current state for validity.
        if cs.post_unify_check(S.data):
            yield S

    return neq_goal


class PredicateStore(ConstraintStore, ABC):
    """An abstract store for testing simple predicates."""

    # Require that all constraints be satisfied for a term; otherwise, succeed
    # if only one is satisfied.
    require_all_constraints = True

    # @abstractmethod
    # def cterm_type_check(self, lvt):
    #     """Check the type of the constrained term when it's ground."""
    #     raise NotImplementedError()

    @abstractmethod
    def cparam_type_check(self, lvt):
        """Check the type of the constraint parameter when it's ground."""
        raise NotImplementedError()

    @abstractmethod
    def constraint_check(self, lv, lvt):
        """Check the constrained term against the constraint parameters when they're ground.

        I.e. test the constraint.
        """
        raise NotImplementedError()

    @abstractmethod
    def constraint_isground(self, lv, lvar_map):
        """Check whether or not the constrained term is "ground enough" to be checked."""  # noqa: E501
        raise NotImplementedError()

    def post_unify_check(self, lvar_map, lvar=None, value=None, old_state=None):

        for lv_key, constraints in list(self.lvar_constraints.items()):

            lv = reify(lv_key, lvar_map)

            is_lv_ground = self.constraint_isground(lv, lvar_map) or isground(
                lv, lvar_map
            )

            if not is_lv_ground:
                # This constraint isn't ready to be checked
                continue

            # if is_lv_ground and not self.cterm_type_check(lv):
            #     self.lvar_constraints[lv_key]
            #     return False

            constraint_grps = groupby(
                lambda x: isground(x, lvar_map), reify(iter(constraints), lvar_map)
            )

            constraints_unground = constraint_grps.get(False, ())
            constraints_ground = constraint_grps.get(True, ())

            if len(constraints_ground) > 0 and not all(
                self.cparam_type_check(c) for c in constraints_ground
            ):
                # Some constraint parameters aren't the correct type, so fail.
                # del self.lvar_constraints[lv_key]
                return False

            assert constraints_unground or constraints_ground

            if is_lv_ground and len(constraints_unground) == 0:

                if self.require_all_constraints and any(
                    not self.constraint_check(lv, t) for t in constraints_ground
                ):
                    return False
                elif not self.require_all_constraints and not any(
                    self.constraint_check(lv, t) for t in constraints_ground
                ):
                    return False

                # The instance and constraint parameters are all ground and the
                # constraint is satisfied, so, since nothing should change from
                # here on, we can remove the constraint.

                del self.lvar_constraints[lv_key]

        # Some types are unground, so we continue checking until they are
        return True

    def pre_unify_check(self, lvar_map, lvar=None, value=None):
        return True


class TypeStore(PredicateStore):
    """A constraint store for asserting object types."""

    require_all_constraints = True

    op_str = "typeo"

    def __init__(self, lvar_constraints=None):
        super().__init__(lvar_constraints)

    def add(self, lvt, cparams):
        if lvt in self.lvar_constraints:
            raise ValueError("Only one type constraint can be applied to a term")

        return super().add(lvt, cparams)

    # def cterm_type_check(self, lvt):
    #     return True

    def cparam_type_check(self, x):
        return isinstance(x, type)

    def constraint_check(self, x, cx):
        return type(x) == cx

    def constraint_isground(self, lv, lvar_map):
        return not (isinstance(lv, Var) or issubclass(type(lv), ConsPair))


def typeo(u, u_type):
    """Construct a goal specifying the type of a term."""

    def typeo_goal(S):
        nonlocal u, u_type

        u_rf, u_type_rf = reify((u, u_type), S)

        if not isground(u_rf, S) or not isground(u_type_rf, S):

            if not isinstance(S, ConstrainedState):
                S = ConstrainedState(S)

            cs = S.constraints.setdefault(TypeStore, TypeStore())

            try:
                cs.add(u_rf, u_type_rf)
            except TypeError:
                # If the instance object can't be hashed, we can simply use a
                # logic variable to uniquely identify it.
                u_lv = var()
                S[u_lv] = u_rf
                cs.add(u_lv, u_type_rf)

            if cs.post_unify_check(S.data, u_rf, u_type_rf):
                yield S

        elif isinstance(u_type_rf, type) and type(u_rf) == u_type_rf:
            yield S

    return typeo_goal


class IsinstanceStore(PredicateStore):
    """A constraint store for asserting object instance types."""

    op_str = "isinstanceo"

    # Satisfying any one constraint is good enough
    require_all_constraints = False

    def __init__(self, lvar_constraints=None):
        super().__init__(lvar_constraints)

    # def cterm_type_check(self, lvt):
    #     return True

    def cparam_type_check(self, lvt):
        return isinstance(lvt, type)

    def constraint_check(self, lv, lvt):
        return isinstance(lv, lvt)

    def constraint_isground(self, lv, lvar_map):
        return not (isinstance(lv, Var) or issubclass(type(lv), ConsPair))


def isinstanceo(u, u_type):
    """Construct a goal specifying that a term is an instance of a type.

    Only a single instance type can be assigned per goal, i.e.

        lany(isinstanceo(var(), list),
             isinstanceo(var(), tuple))

    and not

        isinstanceo(var(), (list, tuple))

    """

    def isinstanceo_goal(S):
        nonlocal u, u_type

        u_rf, u_type_rf = reify((u, u_type), S)

        if not isground(u_rf, S) or not isground(u_type_rf, S):

            if not isinstance(S, ConstrainedState):
                S = ConstrainedState(S)

            cs = S.constraints.setdefault(IsinstanceStore, IsinstanceStore())

            try:
                cs.add(u_rf, u_type_rf)
            except TypeError:
                # If the instance object can't be hashed, we can simply use a
                # logic variable to uniquely identify it.
                u_lv = var()
                S[u_lv] = u_rf
                cs.add(u_lv, u_type_rf)

            if cs.post_unify_check(S.data, u_rf, u_type_rf):
                yield S

        # elif isground(u_type, S):
        #     yield from lany(eq(u_type, u_t) for u_t in type(u).mro())(S)
        elif (
            isinstance(u_type_rf, type)
            # or (
            #     isinstance(u_type, Iterable)
            #     and all(isinstance(t, type) for t in u_type)
            # )
        ) and isinstance(u_rf, u_type_rf):
            yield S

    return isinstanceo_goal



from typing import Callable, Tuple

class FunctionConstraintStore(PredicateStore):
    op_str = "foreigno"
    require_all_constraints = True  # alle Constraints für den Term müssen wahr sein

    def cparam_type_check(self, p):
        # Callable oder ein (callable, extra_args_tuple)
        if callable(p):
            return True
        if isinstance(p, tuple) and len(p) == 2 and callable(p[0]) and isinstance(p[1], tuple):
            return True
        return False
    
    def constraint_isground(self, lv, lvar_map):
        return isground(lv, lvar_map)

    def constraint_check(self, lv, pred):
        if callable(pred):
            func, extra = pred, ()
        else:
            func, extra = pred[0], pred[1]

        args = tuple(lv) if isinstance(lv, (tuple, list)) else (lv,)

        try:
            out = func(*args, *extra)
        except Exception:
            return False  # strict: Fehler == Constraint verletzt

        # strict Bool: nur True akzeptieren
        return True if out is True else False
    '''
    def post_unify_check(self, lvar_map, lvar=None, value=None, old_state=None):
        for lv_key, constraints in list(self.lvar_constraints.items()):
            lv = reify(lv_key, lvar_map)

            if not (self.constraint_isground(lv, lvar_map)):
                # DEBUG
                print("[STORE] skip (not ground):", type(lv), "constraints:", len(constraints))
                continue

            constraints_rf = reify(tuple(constraints), lvar_map)
            results = []
            undecided = False
            for pr in constraints_rf:
                r = self.constraint_check(lv, pr)
                # DEBUG
                # print("[STORE] check →", r)
                if r is None:
                    undecided = True
                results.append(r)

            if not undecided:
                if self.require_all_constraints and any(r is False for r in results):
                    # print("[STORE] → FAIL (False result)")
                    return False
                if (not self.require_all_constraints) and not any(r is True for r in results):
                    # print("[STORE] → FAIL (no True among results)")
                    return False
                # alles entschieden & erfüllt → löschen
                # print("[STORE] → satisfied, remove key")
                del self.lvar_constraints[lv_key]
            else:
                # print("[STORE] undecided → keep constraint")
                pass

        return True
    '''
def foreigno(func: Callable, *terms, extra_args: Tuple=()):
    def goal(S):
        u_terms = reify(terms, S)

        if all(isground(t, S) for t in u_terms):
            try:
                ok = func(*u_terms)
            except Exception:
                return
            if ok is True:
                yield S
            return

        # Falls S noch kein ConstrainedState ist: upgraden
        if not isinstance(S, ConstrainedState):
            S = ConstrainedState(S)

        cs = S.constraints.setdefault(FunctionConstraintStore, FunctionConstraintStore())
        constraint = (func, tuple(extra_args)) if extra_args else func
        try:
            # TERME-Tupel als Key (mehrstellig!)
            cs.add(u_terms, constraint)
        except TypeError:
            # Falls unhashbar: frische Logikvar als Stellvertreter
            u_lv = var()
            S[u_lv] = u_terms
            cs.add(u_lv, constraint)

        
        res = cs.post_unify_check(S.data, u_terms, constraint)
        if res is False:
            return  # Constraint verletzt → prune
        # res ist True oder None: yield S (weiterführen oder delay)
        yield S
    return goal

########################## test
from typing import Callable, Tuple, Iterable

class BindingFunctionConstraintStore(PredicateStore):
    op_str = "foreigno_bind"
    require_all_constraints = True  # alle Constraints zu einem Key müssen „ok“ sein

    def cparam_type_check(self, p):
        # callable oder (callable, extra_args_tuple)
        return callable(p) or (isinstance(p, tuple) and len(p) == 2 and callable(p[0]) and isinstance(p[1], tuple))
    
    def constraint_isground(self, lv, lvar_map):
        # Für Tripel (a, b, houses): „bereit“, sobald die Häuserstruktur (5er-Sequenz) steht
        if isinstance(lv, tuple) and len(lv) == 3:
            _, _, hs = reify(lv, lvar_map)
            try:
                return len(hs) == 5  # Struktur vorhanden → wir dürfen prüfen/binden
            except TypeError:
                return False
        # Fallback
        return isground(lv, lvar_map)
    
    def constraint_check(self, lv, lvt):
        kind, payload = self._run_pred(lv, lvt)
        if kind == "ok":
            return payload is True
        return True

    def _run_pred(self, lv, pred):
        if callable(pred):
            func, extra = pred, ()
        else:
            func, extra = pred[0], pred[1]

        args = tuple(lv) if isinstance(lv, (tuple, list)) else (lv,)

        try:
            out = func(*args, *extra)
        except Exception:
            return ("ok", False)  # Fehler ⇒ Constraint verletzt

        if out is True or out is False or out is None:
            return ("ok", out)

        # Bindungsprotokoll: ("bind", list_of_pairs)
        if isinstance(out, tuple) and len(out) == 2 and out[0] == "bind" and isinstance(out[1], Iterable):
            pairs = list(out[1])
            return ("bind", pairs)

        # unbekanntes Format ⇒ verletzen
        return ("ok", False)

    def post_unify_check(self, lvar_map, lvar=None, value=None, old_state=None):

        ####test
        if getattr(self, "_reentrant", False):
            return True

        for lv_key, constraints in list(self.lvar_constraints.items()):
            lv = reify(lv_key, lvar_map)

            if not self.constraint_isground(lv, lvar_map):
                continue

            constraints_rf = reify(tuple(constraints), lvar_map)
            pending_binds = []
            undecided = False

            for pr in constraints_rf:
                kind, payload = self._run_pred(lv, pr)

                if kind == "ok":
                    if payload is False:
                        return False         # harter Verstoß
                    if payload is None:
                        undecided = True     # noch nicht entscheidbar
                    # True → nichts weiter zu tun
                elif kind == "bind":
                    # deterministische Gleichungen sammeln
                    pending_binds.extend(payload)

            # Falls es Bindungen gibt: anwenden (deterministisch, ohne zu verzweigen)
            if pending_binds:
                # (1) Constraint zuerst deaktivieren, damit Re-Entrant-Aufrufe es nicht nochmal sehen
                if lv_key in self.lvar_constraints:
                    del self.lvar_constraints[lv_key]

                try:
                    # (2) Reentrancy-Guard setzen
                    self._reentrant = True

                    S2 = old_state
                    for (u, v) in pending_binds:
                        S2 = unify(u, v, S2)        # ruft intern wieder post_unify_check auf
                        if S2 is False:
                            return False

                    # (3) Änderungen in-place zurückspielen
                    old_state.data.clear()
                    old_state.data.update(S2.data)
                    old_state.constraints = S2.constraints

                finally:
                    self._reentrant = False

                continue

        return True

def foreigno_bind(func: Callable, *terms, extra_args: Tuple = ()):
    def goal(S):
        u_terms = reify(terms, S)
        if not isinstance(S, ConstrainedState):
            S = ConstrainedState(S)

        cs = S.constraints.setdefault(BindingFunctionConstraintStore, BindingFunctionConstraintStore())
        constraint = (func, tuple(extra_args)) if extra_args else func
        try:
            cs.add(u_terms, constraint)
        except TypeError:
            u_lv = var()
            S[u_lv] = u_terms
            cs.add(u_lv, constraint)

        ok = cs.post_unify_check(S.data, u_terms, constraint, old_state=S)
        if ok is False:
            return
        yield S
    return goal

class UnaryFunctionStore(PredicateStore):
    op_str = "foreignu"
    require_all_constraints = True

    def cparam_type_check(self, p):
        return callable(p)

    def constraint_isground(self, lv, lvar_map):
        return isground(lv, lvar_map)

    def constraint_check(self, lv, pred):
        try:
            out = pred(lv)  # lv ist atomar ODER ein Tupel
        except Exception:
            return False
        return True if out is True else False

from functools import partial
from unification import var, reify

def foreignu(pred_unary, term):
    def goal(S):
        u_term = reify(term, S)
        print("[foreignu] u_term:", u_term)
        # Fast-Path: sofort prüfen, wenn ground
        if isground(u_term, S):
            try:
                if pred_unary(u_term) is True:
                    yield S
                # False/Exception => fail
            except Exception:
                return
            return

        # sonst defer: in Store eintragen und später prüfen
        if not isinstance(S, ConstrainedState):
            S = ConstrainedState(S)

        cs = S.constraints.setdefault(UnaryFunctionStore, UnaryFunctionStore())
        try:
            cs.add(u_term, pred_unary)
        except TypeError:
            u_lv = var()
            S[u_lv] = u_term
            cs.add(u_lv, pred_unary)

        if cs.post_unify_check(S.data, u_term, pred_unary, old_state=S):
            yield S
    return goal

from kanren import conde, eq, lall, membero, run, lany

def foreignn(fn, *terms):
    """Wrappt fn(x1, ..., xN) zu pred_unary(lv_tuple) und ruft foreignu."""
    packed = tuple(terms)

    def pred_unary(lv):
        if not isinstance(lv, tuple):
            lv = (lv,)
        return fn(*lv)

    return foreignu(pred_unary, packed)

def nexto_rel_from_bool(nexto_bool, a, b, hs, n=5):
    """Hebt nexto_bool zu einer Relation an."""
    cases = []
    for i in range(n - 1):
        cases.append(lall(eq(hs[i], a), eq(hs[i+1], b),
                          foreignn(nexto_bool, a, b, hs)))
        cases.append(lall(eq(hs[i], b), eq(hs[i+1], a),
                          foreignn(nexto_bool, a, b, hs)))
    return lany(*cases)

class UnaryBindingFunctionStore(PredicateStore):
    """
    Ein Store für unäre (ein-Argument) FFI-Prädikate, die neben True/False/None
    auch Bindungen ausführen können, per Protokoll:
       pred_unary(lv) -> ("bind", [(u,v), ...]) | True | False | None
    """
    op_str = "foreignu_bind"
    require_all_constraints = True

    # ----------- Typ-Check für Constraint-Parameter --------------------------
    def cparam_type_check(self, p):
        return callable(p)

    # ----------- Groundness: "bereit für Check?" -----------------------------
    def constraint_isground(self, lv, lvar_map):
        # Optional: domänenspezifische Groundness (z.B. Zebra: (a,b,hs) mit len(hs)==5)
        if isinstance(lv, tuple) and len(lv) == 3:
            _, _, hs = reify(lv, lvar_map)
            try:
                return len(hs) == 5
            except TypeError:
                return False
        # Default: voll ground
        return isground(lv, lvar_map)

    # ----------- Einzel-Check eines Constraints ------------------------------
    def constraint_check(self, lv, pred_unary):
        """
        pred_unary(lv) -> ("bind", pairs) | True | False | None
        Rückgabe:
          - True  -> erfüllt
          - False -> Verstoß
          - None  -> (noch) unentscheidbar
          - ("bind", pairs) -> Bindungen anwenden (deterministisch) und als erfüllt zählen
        """
        try:
            out = pred_unary(lv)
        except Exception:
            return False  # Exceptions als Verstoß werten

        # nur OK / DEFER, Bindungen werden in post_unify_check ausgeführt
        if out is True or out is False or out is None:
            return out
        if isinstance(out, tuple) and len(out) == 2 and out[0] == "bind" and isinstance(out[1], Iterable):
            # Speziell kennzeichnen: post_unify_check wird sie ausführen
            return ("bind", list(out[1]))
        return False  # unbekanntes Format => Verstoß

    # ----------- Sweep + Bindungen anwenden ----------------------------------
    def post_unify_check(self, lvar_map, lvar=None, value=None, old_state=None):
        # Reentrancy-Guard (wichtig, weil unify erneut hierher führt)
        if getattr(self, "_reentrant", False):
            return True

        for lv_key, constraints in list(self.lvar_constraints.items()):
            lv = reify(lv_key, lvar_map)
            if not self.constraint_isground(lv, lvar_map):
                continue

            constraints_rf = reify(tuple(constraints), lvar_map)
            pending_binds = []
            undecided = False

            for pred_unary in constraints_rf:
                res = self.constraint_check(lv, pred_unary)
                if res is False:
                    return False
                if res is None:
                    undecided = True
                elif isinstance(res, tuple) and res[0] == "bind":
                    pending_binds.extend(res[1])

            # Bindungen deterministisch anwenden (wenn vorhanden)
            if pending_binds:
                if lv_key in self.lvar_constraints:
                    del self.lvar_constraints[lv_key]
                try:
                    self._reentrant = True
                    S2 = old_state
                    for (u, v) in pending_binds:
                        S2 = unify(u, v, S2)
                        if S2 is False:
                            return False
                    # Änderungen zurück in den alten State schreiben
                    old_state.data.clear()
                    old_state.data.update(S2.data)
                    old_state.constraints = S2.constraints
                finally:
                    self._reentrant = False
                # nach Bindungen: weiter mit nächstem Key (die Bindungen könnten neue Checks triggern)
                continue

            # alle Constraints ok/none und keine Bindungen: nichts zu tun
        return True
    
# Allgemeiner unärer, bindender Store (domain-agnostisch)
class UnaryBindingFunctionStore(PredicateStore):
    op_str = "foreignu_bind"
    require_all_constraints = True

    def cparam_type_check(self, p):
        # Wir speichern direkt ein unäres Callable
        return callable(p)

    def constraint_isground(self, lv, lvar_map):
        # Voll allgemein: prüfen erst, wenn lv (z. B. Tupel) ground ist
        return isground(lv, lvar_map)

    def constraint_check(self, lv, pred_unary):
        """
        pred_unary(lv) -> True | False | None | ("bind", [(u,v), ...])
        - True  => erfüllt
        - False => Verstoß
        - None  => (noch) unentscheidbar, deferred
        - ("bind", pairs) => Bindungen werden in post_unify_check angewendet
        """
        try:
            out = pred_unary(lv)
        except Exception:
            return False

        if out is True or out is False or out is None:
            return out

        if (isinstance(out, tuple) and len(out) == 2 and out[0] == "bind"
                and isinstance(out[1], Iterable)):
            # Signal an post_unify_check: es gibt Bindungen auszuführen
            return ("bind", list(out[1]))

        # Unbekanntes Format → Verstoß
        return False

    def post_unify_check(self, lvar_map, lvar=None, value=None, old_state=None):
        if getattr(self, "_reentrant", False):
            return True

        for lv_key, constraints in list(self.lvar_constraints.items()):
            lv = reify(lv_key, lvar_map)
            if not self.constraint_isground(lv, lvar_map):
                continue

            constraints_rf = reify(tuple(constraints), lvar_map)
            pending_binds = []
            undecided = False

            for pred_unary in constraints_rf:
                res = self.constraint_check(lv, pred_unary)
                if res is False:
                    return False
                if res is None:
                    undecided = True
                elif isinstance(res, tuple) and res[0] == "bind":
                    pending_binds.extend(res[1])

            if pending_binds:
                # Key temporär entfernen, Reentrancy-Guard setzen
                if lv_key in self.lvar_constraints:
                    del self.lvar_constraints[lv_key]
                try:
                    self._reentrant = True
                    S2 = old_state
                    for (u, v) in pending_binds:
                        S2 = unify(u, v, S2)
                        if S2 is False:
                            return False
                    # Änderungen zurückspielen
                    old_state.data.clear()
                    old_state.data.update(S2.data)
                    old_state.constraints = S2.constraints
                finally:
                    self._reentrant = False
                continue  # nächster Key

        return True

def foreignu_bind(pred_unary, term):
    """
    pred_unary: Callable[[lv], True | False | None | ("bind", pairs)]
    term:       lv (oft ein Tupel wie (a,b,context))
    """
    def goal(S):
        u_term = reify(term, S)

        # Fast-Path
        if isground(u_term, S):
            out = pred_unary(u_term)
            if out is False or out is None:
                return
            if out is True:
                yield S; return
            if isinstance(out, tuple) and len(out) == 2 and out[0] == "bind":
                S2 = S
                for u, v in out[1]:
                    S2 = unify(u, v, S2)
                    if S2 is False:
                        return
                yield S2; return
            return

        # Deferred
        if not isinstance(S, ConstrainedState):
            S = ConstrainedState(S)

        cs = S.constraints.setdefault(UnaryBindingFunctionStore, UnaryBindingFunctionStore())
        try:
            cs.add(u_term, pred_unary)
        except TypeError:
            u_lv = var(); S[u_lv] = u_term; cs.add(u_lv, pred_unary)

        ok = cs.post_unify_check(S.data, u_term, pred_unary, old_state=S)
        if ok is False:
            return
        yield S
    return goal

    
def all_constraints_resolved():
    def goal(S):
        stores = []
        cs = getattr(S, "constraints", {})
        # ALLE relevanten Stores einsammeln:
        from kanren.constraints import (
            FunctionConstraintStore, BindingFunctionConstraintStore, UnaryFunctionStore
        )
        for K in (FunctionConstraintStore, BindingFunctionConstraintStore, UnaryFunctionStore):
            st = cs.get(K)
            if st is not None:
                stores.append(st)

        # einmal sweepen:
        for st in stores:
            if st.post_unify_check(S.data, old_state=S) is False:
                return  # Widerspruch

        from unification.core import isground
        for st in stores:
            for lv_key in list(st.lvar_constraints.keys()):
                lv = reify(lv_key, S.data)
                if st.constraint_isground(lv, S.data) or isground(lv, S.data):
                    return      
        yield S
    return goal
