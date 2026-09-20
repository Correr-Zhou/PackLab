from runtime.episode_control import EpisodeControl, EpisodeControlConfig


def _control(object_count=5):
    return EpisodeControl(EpisodeControlConfig(max_invalid_actions=3, max_turns_buffer=3, hard_max_turns=40), object_count)


def test_cumulative_invalid_actions_stop_episode():
    control = _control()
    assert not control.update(valid=False, buffer_exhausted=False).done
    assert not control.update(valid=True, buffer_exhausted=False).done
    assert not control.update(valid=False, buffer_exhausted=False).done
    result = control.update(valid=False, buffer_exhausted=False)
    assert result.done
    assert result.termination_reason == "max_invalid_actions"
    assert control.total_invalid_actions == 3


def test_completed_when_buffer_exhausted():
    control = _control()
    result = control.update(valid=True, buffer_exhausted=True)
    assert result.done
    assert result.termination_reason == "completed"


def test_max_turns_exceeded():
    control = EpisodeControl(
        EpisodeControlConfig(max_invalid_actions=100, max_turns_buffer=1, hard_max_turns=4), object_count=2
    )
    assert not control.update(valid=True, buffer_exhausted=False).done
    assert not control.update(valid=True, buffer_exhausted=False).done
    result = control.update(valid=True, buffer_exhausted=False)
    assert result.done
    assert result.termination_reason == "max_turns_exceeded"
