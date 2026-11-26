from sqlalchemy import BigInteger, JSON, MetaData, Text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase


PortableJSON = JSON(none_as_null=True).with_variant(
    postgresql.JSONB(none_as_null=True), "postgresql"
)
PortableJSONValue = JSON().with_variant(postgresql.JSONB(), "postgresql")
PortableTextArray = JSON().with_variant(postgresql.ARRAY(Text), "postgresql")
PortableBigIntArray = JSON().with_variant(postgresql.ARRAY(BigInteger), "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
