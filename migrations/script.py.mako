"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""

from alembic import op

revision: str = "${up_revision}"
down_revision: str | None = ${f'"{down_revision}"' if down_revision else None}
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        """
    )


def downgrade() -> None:
    raise NotImplementedError
