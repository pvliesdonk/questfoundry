"""Tests for Message dataclass and factory functions."""

from datetime import datetime

from questfoundry.runtime.messaging import (
    Message,
    MessagePriority,
    MessageStatus,
    MessageType,
    create_delegation_request,
    create_delegation_response,
    create_digest,
    create_escalation,
    create_feedback,
    create_message,
    create_progress_update,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_basic_message(self):
        """Test creating a basic message."""
        message = Message(
            id="test-id",
            type=MessageType.FEEDBACK,
            from_agent="showrunner",
            to_agent="scene_smith",
            timestamp=datetime.now(tz=None),
            payload={"content": "test"},
        )

        assert message.id == "test-id"
        assert message.type == MessageType.FEEDBACK
        assert message.from_agent == "showrunner"
        assert message.to_agent == "scene_smith"
        assert message.payload == {"content": "test"}
        assert message.status == MessageStatus.PENDING
        assert message.priority == MessagePriority.NORMAL

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ts = datetime(2025, 1, 15, 12, 0, 0)
        message = Message(
            id="test-id",
            type=MessageType.DELEGATION_REQUEST,
            from_agent="showrunner",
            to_agent="plotwright",
            timestamp=ts,
            payload={"task": "create outline"},
            correlation_id="corr-123",
            priority=MessagePriority.HIGH,
        )

        data = message.to_dict()

        assert data["id"] == "test-id"
        assert data["type"] == "delegation_request"
        assert data["from_agent"] == "showrunner"
        assert data["to_agent"] == "plotwright"
        assert data["timestamp"] == "2025-01-15T12:00:00"
        assert data["payload"] == {"task": "create outline"}
        assert data["correlation_id"] == "corr-123"
        assert data["priority"] == MessagePriority.HIGH
        assert data["status"] == "pending"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test-id",
            "type": "feedback",
            "from_agent": "gatekeeper",
            "to_agent": "scene_smith",
            "timestamp": "2025-01-15T12:00:00",
            "payload": {"feedback_type": "style"},
            "priority": 3,
            "status": "delivered",
        }

        message = Message.from_dict(data)

        assert message.id == "test-id"
        assert message.type == MessageType.FEEDBACK
        assert message.from_agent == "gatekeeper"
        assert message.to_agent == "scene_smith"
        assert message.payload == {"feedback_type": "style"}
        assert message.priority == 3
        assert message.status == MessageStatus.DELIVERED

    def test_is_expired_no_ttl(self):
        """Test that messages without TTL never expire."""
        message = Message(
            id="test-id",
            type=MessageType.FEEDBACK,
            from_agent="a",
            to_agent="b",
            timestamp=datetime.now(tz=None),
            ttl_turns=None,
            turn_created=1,
        )

        assert not message.is_expired(100)
        assert not message.is_expired(1000)

    def test_is_expired_with_ttl(self):
        """Test TTL expiration logic."""
        message = Message(
            id="test-id",
            type=MessageType.PROGRESS_UPDATE,
            from_agent="a",
            to_agent="b",
            timestamp=datetime.now(tz=None),
            ttl_turns=5,
            turn_created=10,
        )

        assert not message.is_expired(10)  # Same turn
        assert not message.is_expired(14)  # 4 turns later
        assert not message.is_expired(15)  # 5 turns later (boundary)
        assert message.is_expired(16)  # 6 turns later (expired)
        assert message.is_expired(100)  # Way past

    def test_delegation_request_never_expires(self):
        """Test that delegation requests NEVER expire regardless of TTL."""
        message = Message(
            id="test-id",
            type=MessageType.DELEGATION_REQUEST,
            from_agent="showrunner",
            to_agent="plotwright",
            timestamp=datetime.now(tz=None),
            ttl_turns=1,  # Very short TTL
            turn_created=1,
        )

        # Even with TTL=1 and way past that, delegations don't expire
        assert not message.is_expired(100)
        assert not message.is_expired(1000)

    def test_delegation_response_never_expires(self):
        """Test that delegation responses NEVER expire regardless of TTL."""
        message = Message(
            id="test-id",
            type=MessageType.DELEGATION_RESPONSE,
            from_agent="plotwright",
            to_agent="showrunner",
            timestamp=datetime.now(tz=None),
            ttl_turns=1,
            turn_created=1,
        )

        assert not message.is_expired(100)
        assert not message.is_expired(1000)


class TestCreateMessage:
    """Tests for create_message factory function."""

    def test_creates_message_with_uuid(self):
        """Test that created messages get unique IDs."""
        msg1 = create_message(
            MessageType.FEEDBACK,
            from_agent="a",
            to_agent="b",
        )
        msg2 = create_message(
            MessageType.FEEDBACK,
            from_agent="a",
            to_agent="b",
        )

        assert msg1.id != msg2.id
        assert len(msg1.id) == 36  # UUID format

    def test_default_priority_for_escalation(self):
        """Test escalations get high priority by default."""
        msg = create_message(
            MessageType.ESCALATION,
            from_agent="agent",
            to_agent="showrunner",
        )

        assert msg.priority == MessagePriority.ESCALATION

    def test_default_priority_for_progress(self):
        """Test progress updates get low priority by default."""
        msg = create_message(
            MessageType.PROGRESS_UPDATE,
            from_agent="agent",
            to_agent="showrunner",
        )

        assert msg.priority == MessagePriority.PROGRESS

    def test_explicit_priority_overrides_default(self):
        """Test that explicit priority overrides default."""
        msg = create_message(
            MessageType.ESCALATION,
            from_agent="agent",
            to_agent="showrunner",
            priority=MessagePriority.LOW,
        )

        assert msg.priority == MessagePriority.LOW


class TestDelegationMessages:
    """Tests for delegation factory functions."""

    def test_create_delegation_request(self):
        """Test creating a delegation request."""
        msg = create_delegation_request(
            from_agent="showrunner",
            to_agent="plotwright",
            task="Create story outline",
            context={"genre": "mystery"},
            playbook_id="story_spark",
            turn_created=5,
        )

        assert msg.type == MessageType.DELEGATION_REQUEST
        assert msg.from_agent == "showrunner"
        assert msg.to_agent == "plotwright"
        assert msg.payload["task"] == "Create story outline"
        assert msg.payload["context"] == {"genre": "mystery"}
        assert msg.playbook_id == "story_spark"
        assert msg.turn_created == 5
        assert msg.delegation_id is not None
        assert msg.correlation_id == msg.delegation_id  # Same for linking

    def test_create_delegation_response(self):
        """Test creating a delegation response."""
        msg = create_delegation_response(
            from_agent="plotwright",
            to_agent="showrunner",
            delegation_id="deleg-123",
            success=True,
            result={"outline": "Chapter 1..."},
            artifacts_produced=["artifact-456"],
            in_reply_to="msg-orig",
        )

        assert msg.type == MessageType.DELEGATION_RESPONSE
        assert msg.from_agent == "plotwright"
        assert msg.to_agent == "showrunner"
        assert msg.delegation_id == "deleg-123"
        assert msg.correlation_id == "deleg-123"
        assert msg.in_reply_to == "msg-orig"
        assert msg.payload["success"] is True
        assert msg.payload["result"] == {"outline": "Chapter 1..."}
        assert msg.payload["artifacts_produced"] == ["artifact-456"]

    def test_create_delegation_response_failure(self):
        """Test creating a failed delegation response."""
        msg = create_delegation_response(
            from_agent="plotwright",
            to_agent="showrunner",
            delegation_id="deleg-123",
            success=False,
            error="Missing required context",
        )

        assert msg.payload["success"] is False
        assert msg.payload["error"] == "Missing required context"


class TestEscalationMessages:
    """Tests for escalation factory function."""

    def test_create_escalation(self):
        """Test creating an escalation message."""
        msg = create_escalation(
            from_agent="gatekeeper",
            to_agent="showrunner",
            reason="max_rework_exceeded",
            details="Section failed quality gate 3 times",
            playbook_id="scene_weave",
            phase_id="prose_drafting",
            rework_count=3,
            attempted_resolutions=["Voice fix", "Structure fix", "Content fix"],
            suggested_action="Manual review required",
        )

        assert msg.type == MessageType.ESCALATION
        assert msg.priority == MessagePriority.ESCALATION
        assert msg.payload["reason"] == "max_rework_exceeded"
        assert msg.payload["details"] == "Section failed quality gate 3 times"
        assert msg.payload["rework_count"] == 3
        assert len(msg.payload["attempted_resolutions"]) == 3
        assert msg.payload["suggested_action"] == "Manual review required"
        assert msg.playbook_id == "scene_weave"
        assert msg.phase_id == "prose_drafting"


class TestFeedbackMessages:
    """Tests for feedback factory function."""

    def test_create_feedback(self):
        """Test creating a feedback message."""
        msg = create_feedback(
            from_agent="style_lead",
            to_agent="scene_smith",
            artifact_id="section-123",
            feedback_type="voice",
            content="Consider more active verbs",
            actionable=True,
            severity="minor",
        )

        assert msg.type == MessageType.FEEDBACK
        assert msg.payload["artifact_id"] == "section-123"
        assert msg.payload["feedback_type"] == "voice"
        assert msg.payload["content"] == "Consider more active verbs"
        assert msg.payload["actionable"] is True
        assert msg.payload["severity"] == "minor"


class TestProgressMessages:
    """Tests for progress update factory function."""

    def test_create_progress_update(self):
        """Test creating a progress update."""
        msg = create_progress_update(
            from_agent="scene_smith",
            to_agent="showrunner",
            status="Drafting section 2 of 5",
            progress_pct=40,
            current_step="prose_drafting",
            turn_created=10,
        )

        assert msg.type == MessageType.PROGRESS_UPDATE
        assert msg.priority == MessagePriority.PROGRESS
        assert msg.payload["status"] == "Drafting section 2 of 5"
        assert msg.payload["progress_pct"] == 40
        assert msg.payload["current_step"] == "prose_drafting"
        assert msg.ttl_turns == 5  # Default TTL for progress

    def test_progress_update_custom_ttl(self):
        """Test progress update with custom TTL."""
        msg = create_progress_update(
            from_agent="a",
            to_agent="b",
            status="test",
            ttl_turns=10,
        )

        assert msg.ttl_turns == 10


class TestDigestMessages:
    """Tests for digest message factory function."""

    def test_create_digest_basic(self):
        """Test creating a basic digest message."""
        # Create some original messages to summarize
        original_messages = [
            Message(
                id=f"msg-{i}",
                type=MessageType.PROGRESS_UPDATE,
                from_agent="scene_smith",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, i, 0),
                priority=MessagePriority.PROGRESS,
            )
            for i in range(5)
        ]

        msg = create_digest(
            to_agent="showrunner",
            summary="Summary of 5 progress updates from scene_smith",
            original_messages=original_messages,
            turn_created=10,
        )

        assert msg.type == MessageType.DIGEST
        assert msg.from_agent == "runtime"  # Digests are from runtime
        assert msg.to_agent == "showrunner"
        assert msg.priority == MessagePriority.DIGEST

        payload = msg.payload["digest"]
        assert payload["summary"] == "Summary of 5 progress updates from scene_smith"
        assert payload["summarized_count"] == 5
        assert len(payload["original_ids"]) == 5
        assert "scene_smith" in payload["original_senders"]
        assert payload["contains_delegations"] is False
        # PROGRESS priority (-2) is > LOW (-5), so urgency is "normal"
        assert payload["urgency"] == "normal"

    def test_create_digest_with_delegations(self):
        """Test that digest correctly flags contained delegations."""
        original_messages = [
            Message(
                id="msg-1",
                type=MessageType.DELEGATION_REQUEST,
                from_agent="showrunner",
                to_agent="plotwright",
                timestamp=datetime(2025, 1, 15, 12, 0, 0),
                priority=MessagePriority.DELEGATION,
            ),
            Message(
                id="msg-2",
                type=MessageType.PROGRESS_UPDATE,
                from_agent="plotwright",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, 1, 0),
                priority=MessagePriority.PROGRESS,
            ),
        ]

        msg = create_digest(
            to_agent="showrunner",
            summary="Summary with delegation",
            original_messages=original_messages,
        )

        payload = msg.payload["digest"]
        assert payload["contains_delegations"] is True

    def test_create_digest_with_delegation_response(self):
        """Test that digest correctly flags DELEGATION_RESPONSE as containing delegations."""
        original_messages = [
            Message(
                id="msg-1",
                type=MessageType.DELEGATION_RESPONSE,
                from_agent="plotwright",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, 0, 0),
                priority=MessagePriority.DELEGATION,
            ),
        ]

        msg = create_digest(
            to_agent="showrunner",
            summary="Summary with delegation response",
            original_messages=original_messages,
        )

        payload = msg.payload["digest"]
        assert payload["contains_delegations"] is True

    def test_create_digest_empty_messages_raises(self):
        """Test that create_digest raises ValueError for empty message list."""
        import pytest

        with pytest.raises(ValueError, match="at least one message"):
            create_digest(
                to_agent="showrunner",
                summary="Summary",
                original_messages=[],
            )

    def test_create_digest_with_high_priority(self):
        """Test that digest correctly captures urgency from high priority messages."""
        original_messages = [
            Message(
                id="msg-1",
                type=MessageType.ESCALATION,
                from_agent="gatekeeper",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, 0, 0),
                priority=MessagePriority.ESCALATION,  # High priority
            ),
        ]

        msg = create_digest(
            to_agent="showrunner",
            summary="Summary with escalation",
            original_messages=original_messages,
        )

        payload = msg.payload["digest"]
        assert payload["urgency"] == "high"

    def test_create_digest_with_action_items(self):
        """Test that action items are included in digest."""
        original_messages = [
            Message(
                id="msg-1",
                type=MessageType.FEEDBACK,
                from_agent="style_lead",
                to_agent="showrunner",
                timestamp=datetime(2025, 1, 15, 12, 0, 0),
            ),
        ]

        action_items = ["Review style feedback", "Address voice consistency"]

        msg = create_digest(
            to_agent="showrunner",
            summary="Summary",
            original_messages=original_messages,
            action_items=action_items,
        )

        payload = msg.payload["digest"]
        assert payload["action_items"] == action_items

    def test_create_digest_time_range(self):
        """Test that digest captures time range of original messages."""
        original_messages = [
            Message(
                id="msg-1",
                type=MessageType.FEEDBACK,
                from_agent="a",
                to_agent="b",
                timestamp=datetime(2025, 1, 15, 10, 0, 0),  # Earlier
            ),
            Message(
                id="msg-2",
                type=MessageType.FEEDBACK,
                from_agent="c",
                to_agent="b",
                timestamp=datetime(2025, 1, 15, 14, 0, 0),  # Later
            ),
        ]

        msg = create_digest(
            to_agent="b",
            summary="Summary",
            original_messages=original_messages,
        )

        payload = msg.payload["digest"]
        assert payload["time_range"]["earliest"] == "2025-01-15T10:00:00"
        assert payload["time_range"]["latest"] == "2025-01-15T14:00:00"
