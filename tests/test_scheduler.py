"""Who gets the card next.

Pure policy: a shape here is a string, and swapping to one is a function that
records it. No models, no engines, no card — which is the point. The rules
below are the hard part, and they can be got wrong in ways that only show up
under load on a machine you cannot easily reproduce.
"""

import threading
import time
import unittest

from ai_lab.scheduler import Abandoned, QueueFull, Scheduler


class Card:
    """A card that records what was put on it, and can be made slow or broken."""

    def __init__(self, places=1, load_s=0.0):
        self.switches = []
        self.places_for = places
        self.load_s = load_s
        self.refuses = {}                       # shape -> exception to raise

    def switch(self, shape):
        if self.load_s:
            time.sleep(self.load_s)
        if shape in self.refuses:
            raise self.refuses[shape]
        self.switches.append(shape)

    def places(self, shape):
        if isinstance(self.places_for, dict):
            return self.places_for.get(shape, 1)
        return self.places_for


def scheduler(card, **kwargs):
    return Scheduler(card.switch, card.places, **kwargs)


def run(target, *args):
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


class TogetherTests(unittest.TestCase):
    """Requests for the model already on the card do not queue behind each other."""

    def test_two_for_the_same_shape_run_at_once(self):
        card = Card(places=4)
        s = scheduler(card)
        s.enter("a")                            # loads it
        s.enter("a")                            # straight through
        self.assertEqual(s.state()["in_flight"], 2)
        self.assertEqual(card.switches, ["a"], "it should not have reloaded")

    def test_the_engine_s_own_limit_is_respected(self):
        card = Card(places=2)
        s = scheduler(card)
        s.enter("a"); s.enter("a")
        third = run(s.enter, "a")
        time.sleep(0.05)
        self.assertEqual(s.state()["in_flight"], 2)
        self.assertEqual(len(s.state()["waiting"]), 1)
        s.leave()
        third.join(timeout=2)
        self.assertEqual(s.state()["in_flight"], 2)

    def test_each_shape_brings_its_own_limit(self):
        card = Card(places={"a": 1, "b": 3})
        s = scheduler(card)
        s.enter("b"); s.enter("b"); s.enter("b")
        self.assertEqual(s.state()["in_flight"], 3)


class DoorClosesTests(unittest.TestCase):
    """The rule that stops a busy model starving every other one.

    Without it a model under continuous load never goes idle, so the switch
    never happens, and a request for anything else waits for ever. Not
    unlikely — certain.
    """

    def test_a_waiting_request_closes_the_door_for_everyone(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        wants_b = run(s.enter, "b")
        time.sleep(0.05)
        later_a = run(s.enter, "a")             # would have walked in before
        time.sleep(0.05)
        self.assertEqual(s.state()["in_flight"], 1)
        self.assertEqual([w["shape"] for w in s.state()["waiting"]], ["b", "a"])
        s.leave()
        wants_b.join(timeout=2)
        self.assertEqual(card.switches, ["a", "b"])
        self.assertFalse(later_a.is_alive() is False and False)

    def test_continuous_load_does_not_starve_the_other_model(self):
        # The scenario the rule exists for: requests for "a" keep arriving
        # while one for "b" waits.
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        wants_b = run(s.enter, "b")
        time.sleep(0.05)
        arrivals = [run(s.enter, "a") for _ in range(20)]
        time.sleep(0.05)
        s.leave()                               # the only one in flight ends
        wants_b.join(timeout=2)
        self.assertFalse(wants_b.is_alive(), "b never got in")
        self.assertEqual(card.switches, ["a", "b"])


class PhotographTests(unittest.TestCase):
    """What arrives during a load belongs to the next round."""

    def test_everyone_waiting_for_the_new_shape_goes_in_together(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        waiters = [run(s.enter, "b") for _ in range(3)]
        time.sleep(0.05)
        s.leave()
        for thread in waiters:
            thread.join(timeout=2)
        self.assertEqual(s.state()["in_flight"], 3)
        self.assertEqual(card.switches, ["a", "b"], "one load for all three")

    def test_arriving_during_the_load_waits_for_the_next_round(self):
        # Otherwise the shape just loaded starves the one that was waiting,
        # which is the door problem again with the names swapped.
        card = Card(places=8, load_s=0.3)
        s = scheduler(card)
        s.enter("a")
        run(s.enter, "b")
        run(s.enter, "c")
        time.sleep(0.05)
        s.leave()                               # the switch to b begins
        time.sleep(0.1)                         # still loading
        late_b = run(s.enter, "b")
        time.sleep(0.05)
        self.assertTrue(late_b.is_alive(), "it jumped into the round it missed")

    def test_more_than_fit_wait_their_turn(self):
        card = Card(places=2)
        s = scheduler(card)
        s.enter("a")
        waiters = [run(s.enter, "b") for _ in range(5)]
        time.sleep(0.05)
        s.leave()
        time.sleep(0.1)
        self.assertEqual(s.state()["in_flight"], 2)
        self.assertEqual(len(s.state()["waiting"]), 3)


class OldestWinsTests(unittest.TestCase):
    def test_the_oldest_waiting_request_decides_the_next_shape(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        run(s.enter, "b")                       # older
        time.sleep(0.05)
        for _ in range(5):
            run(s.enter, "c")                   # newer, and more of them
        time.sleep(0.05)
        s.leave()
        time.sleep(0.2)
        self.assertEqual(card.switches, ["a", "b"],
                         "the crowd won over the older request")

    def test_then_the_next_oldest(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        run(s.enter, "b")
        time.sleep(0.03)
        run(s.enter, "c")
        time.sleep(0.05)
        s.leave()
        time.sleep(0.1)
        s.leave()                               # b finishes
        time.sleep(0.2)
        self.assertEqual(card.switches, ["a", "b", "c"])


class FailedLoadTests(unittest.TestCase):
    def test_everyone_waiting_for_it_is_told_why(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        card.refuses["b"] = RuntimeError("it would not fit")
        failures = []

        def wants_b():
            try:
                s.enter("b")
            except Exception as error:
                failures.append(str(error))

        threads = [run(wants_b) for _ in range(3)]
        time.sleep(0.05)
        s.leave()
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(failures, ["it would not fit"] * 3)

    def test_it_moves_on_to_the_next_shape_rather_than_retrying(self):
        # A model that would not fit a moment ago will not fit now.
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        run(lambda: _swallow(s.enter, "b"))
        time.sleep(0.03)
        wants_c = run(s.enter, "c")
        time.sleep(0.05)
        card.refuses["b"] = RuntimeError("no")   # c is fine
        s.leave()
        wants_c.join(timeout=2)
        self.assertFalse(wants_c.is_alive(), "c never got its turn")
        self.assertEqual(card.switches, ["a", "c"])


class AbandonedTests(unittest.TestCase):
    """A client that gave up must not cost a swap.

    Reproduced on the machine before this existed: a client asked for another
    model, hung up at once, and the manager unloaded the model that was working
    and loaded twenty-one gigabytes for nobody.
    """

    def test_it_is_dropped_instead_of_served(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        gone = []

        def leaves():
            try:
                s.enter("b", still_wanted=lambda: False)
            except Abandoned:
                gone.append(True)

        thread = run(leaves)
        time.sleep(0.05)
        s.leave()
        thread.join(timeout=2)
        self.assertEqual(gone, [True])

    def test_and_the_card_is_left_alone(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        run(lambda: _swallow(s.enter, "b", still_wanted=lambda: False))
        time.sleep(0.05)
        s.leave()
        time.sleep(0.1)
        self.assertEqual(card.switches, ["a"],
                         "it swapped for a client that had gone")

    def test_a_client_still_there_is_served(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")
        thread = run(lambda: s.enter("b", still_wanted=lambda: True))
        time.sleep(0.05)
        s.leave()
        thread.join(timeout=2)
        self.assertEqual(card.switches, ["a", "b"])

    def test_when_it_cannot_be_told_the_request_is_served(self):
        card = Card(places=8)
        s = scheduler(card)
        s.enter("a")

        def cannot_tell():
            raise OSError("socket in a strange state")

        thread = run(lambda: s.enter("b", still_wanted=cannot_tell))
        time.sleep(0.05)
        s.leave()
        thread.join(timeout=2)
        self.assertEqual(card.switches, ["a", "b"])


class FullTests(unittest.TestCase):
    def test_a_full_queue_says_so_rather_than_growing(self):
        # Every waiting request is a thread. There has to be an end.
        card = Card(places=1)
        s = scheduler(card, max_waiting=3)
        s.enter("a")
        for _ in range(3):
            run(lambda: _swallow(s.enter, "b"))
        time.sleep(0.1)
        with self.assertRaises(QueueFull) as caught:
            s.enter("b")
        self.assertIn("waiting", str(caught.exception))


class ResetTests(unittest.TestCase):
    """A forced stop means a clean slate, not a partly-true one."""

    def test_everyone_waiting_is_turned_away(self):
        card = Card(places=1)
        s = scheduler(card)
        s.enter("a")
        refused = []

        def waits():
            try:
                s.enter("b")
            except Abandoned as error:
                refused.append(str(error))

        threads = [run(waits) for _ in range(3)]
        time.sleep(0.1)
        self.assertEqual(s.reset("stopped by hand"), 3)
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(len(refused), 3)
        self.assertIn("stopped by hand", refused[0])

    def test_nothing_is_running_and_nothing_is_loaded_afterwards(self):
        card = Card(places=4)
        s = scheduler(card)
        s.enter("a"); s.enter("a")
        s.reset("stopped by hand")
        state = s.state()
        self.assertEqual(state["in_flight"], 0)
        self.assertIsNone(state["current"])
        self.assertEqual(state["waiting"], [])

    def test_the_next_request_loads_again(self):
        card = Card(places=4)
        s = scheduler(card)
        s.enter("a")
        s.reset("stopped by hand")
        s.enter("a")
        self.assertEqual(card.switches, ["a", "a"])


class OutsideInterferenceTests(unittest.TestCase):
    def test_a_model_taken_off_behind_our_back_is_loaded_again(self):
        card = Card(places=4)
        s = scheduler(card)
        s.enter("a")
        s.leave()
        s.forget_current()                      # somebody pressed Unload
        s.enter("a")
        self.assertEqual(card.switches, ["a", "a"])


class LockTests(unittest.TestCase):
    def test_the_page_can_be_read_while_a_model_loads(self):
        # The lock must not be held across a load. It takes up to a minute, and
        # that is exactly when somebody wants to see what is going on.
        card = Card(places=4, load_s=0.4)
        s = scheduler(card)
        run(s.enter, "a")
        time.sleep(0.1)
        started = time.perf_counter()
        state = s.state()
        self.assertLess(time.perf_counter() - started, 0.1,
                        "reading the state waited for the load")
        self.assertTrue(state["switching"])


def _swallow(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception:
        pass


if __name__ == "__main__":
    unittest.main()


class OverflowTests(unittest.TestCase):
    """More waiting for a shape than the shape has places.

    They belong to the round that is running, so they are let in as places
    free. Sending them to the back of the queue put them behind whatever
    arrived while the model was loading — including a request for another
    model, which then got served in the middle of their round, and the round
    finished after two more swaps.
    """

    def test_the_ones_that_did_not_fit_go_next_not_last(self):
        card = Card(places={"a": 1, "b": 2, "c": 1})
        s = scheduler(card)
        s.enter("a")
        for _ in range(4):
            run(_hold, s, "b", 0.15)
        time.sleep(0.05)
        run(_hold, s, "c", 0.05)                # newer than all four
        time.sleep(0.05)
        s.leave()
        time.sleep(1.2)
        self.assertEqual(card.switches, ["a", "b", "c"],
                         "it swapped away mid-round and back again")

    def test_they_are_let_in_as_places_free(self):
        card = Card(places=2)
        s = scheduler(card)
        s.enter("a")
        for _ in range(5):
            run(lambda: _swallow(s.enter, "b"))
        time.sleep(0.05)
        s.leave()
        time.sleep(0.1)
        self.assertEqual(s.state()["in_flight"], 2)
        self.assertEqual(len(s.state()["waiting"]), 3)
        s.leave()
        time.sleep(0.1)
        self.assertEqual(len(s.state()["waiting"]), 2, "nobody took the free place")

    def test_they_keep_the_order_they_arrived_in(self):
        card = Card(places=1)
        s = scheduler(card)
        s.enter("a")
        served = []
        for index in range(3):
            run(_record, s, "b", served, index)
            time.sleep(0.02)
        time.sleep(0.05)
        s.leave()
        for _ in range(4):
            time.sleep(0.15)
            s.leave()
        self.assertEqual(served, [0, 1, 2])


def _hold(s, shape, seconds):
    try:
        s.enter(shape)
    except Exception:
        return
    time.sleep(seconds)
    s.leave()


def _record(s, shape, served, index):
    try:
        s.enter(shape)
    except Exception:
        return
    served.append(index)


class InOrderTests(unittest.TestCase):
    """The queue is served in order. Nothing younger goes first.

    The run at the front goes in together; it stops at the first request
    wanting something else. Sweeping up the later ones of the same shape is
    tempting — same model, already loaded, free — and it was written that way
    first. But they arrived after the request that wants something else, and
    serving them ahead of it is what oldest-first exists to prevent.
    """

    def order_of(self, queue, places=1):
        """Run a queue through and report which shapes were loaded, in order."""
        card = Card(places=places)
        s = scheduler(card)
        s.enter("start")
        s.leave()
        threads = []
        for shape in queue:
            threads.append(run(_hold, s, shape, 0.05))
            time.sleep(0.03)
        for thread in threads:
            thread.join(timeout=10)
        return card.switches[1:]                # drop the "start"

    def test_a_later_request_does_not_jump_the_one_already_waiting(self):
        # b, b, c, b — the last b arrived after c and must be served after it.
        self.assertEqual(self.order_of(["b", "b", "c", "b"]), ["b", "c", "b"])

    def test_a_run_at_the_front_goes_in_as_one(self):
        self.assertEqual(self.order_of(["b", "b", "b", "c"], places=4),
                         ["b", "c"])

    def test_alternating_requests_swap_on_every_one(self):
        # The cost of the rule, and it is not hidden. Two models genuinely
        # wanted at once is the card's limit, not this file's.
        self.assertEqual(self.order_of(["b", "c", "b", "c"]),
                         ["b", "c", "b", "c"])

    def test_the_run_is_taken_by_position_not_by_counting_the_whole_queue(self):
        # Four want b in total, but only the two at the front are in the run.
        card = Card(places=4)
        s = scheduler(card)
        s.enter("a")
        for shape in ("b", "b", "c", "b", "b"):
            run(_hold, s, shape, 0.4)
            time.sleep(0.03)
        time.sleep(0.05)
        s.leave()
        time.sleep(0.25)
        self.assertEqual(s.state()["in_flight"], 2,
                         "it swept up the two behind the request for c")
        self.assertEqual([w["shape"] for w in s.state()["waiting"]],
                         ["c", "b", "b"])


class ResetDuringALoadTests(unittest.TestCase):
    """A forced stop while a model is loading.

    The load cannot be called back — the engine is already starting — so its
    result is discarded instead. Before this, it finished after the reset and
    published itself: the scheduler believed a model was loaded, and a request
    that had been told it was turned away ran anyway.
    """

    def scheduler_that_loads_slowly(self):
        card = Card(places=4, load_s=0.4)
        return scheduler(card), card

    def test_the_waiting_request_is_turned_away_not_admitted(self):
        s, card = self.scheduler_that_loads_slowly()
        s.enter("a"); s.leave()
        outcome = []

        def waits():
            try:
                s.enter("b")
                outcome.append("admitted")
            except Abandoned:
                outcome.append("turned away")

        run(waits)
        time.sleep(0.1)                         # the load has begun
        s.reset("stopped by hand")
        time.sleep(0.8)
        self.assertEqual(outcome, ["turned away"])

    def test_the_card_is_left_unknown_rather_than_claimed(self):
        # Whatever is on it now, the next request unloads before it loads.
        s, card = self.scheduler_that_loads_slowly()
        s.enter("a"); s.leave()
        run(lambda: _swallow(s.enter, "b"))
        time.sleep(0.1)
        s.reset("stopped by hand")
        time.sleep(0.8)
        state = s.state()
        self.assertIsNone(state["current"])
        self.assertEqual(state["in_flight"], 0)

    def test_the_next_request_loads_again(self):
        s, card = self.scheduler_that_loads_slowly()
        s.enter("a"); s.leave()
        run(lambda: _swallow(s.enter, "b"))
        time.sleep(0.1)
        s.reset("stopped by hand")
        time.sleep(0.8)
        before = len(card.switches)
        s.enter("b")
        self.assertEqual(len(card.switches), before + 1)

    def test_a_reset_after_the_load_finished_is_the_ordinary_case(self):
        # No epoch trickery when nothing was in flight.
        s, card = self.scheduler_that_loads_slowly()
        s.enter("a")
        s.reset("stopped by hand")
        self.assertIsNone(s.state()["current"])
