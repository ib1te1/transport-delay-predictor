"""Empty starting point; the first real schema migration builds on it.

Revision ID: 0001
Revises:
Create Date: 2026-09-22
"""

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
