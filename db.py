import pandas as pd
import taosws
from loguru import logger
from typing import List

STOCK_VALUES = "ts, closing_price, opening_price, highest_price, lowest_price, trading_volume, rise_and_fall"
STOCK_VALUES_CN = "时间戳, 收盘价, 开盘价, 最高价, 最低价, 交易量, 涨跌幅"
STOCK_STABLE = "`financial_data_analysis`.`stock`"


def connection(host, port, user, password, database):
    logger.info(user)
    try:
        conn = taosws.connect(  # type: ignore
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        logger.info(f"Connected to {host}:{port} successfully.")
        return conn
    except Exception as err:
        logger.error(f"Failed to connect to {host}:{port} , ErrMessage:{err}")
        raise err


def query_stock_all_code(conn) -> List[str]:
    return [
        row[0]
        for row in list(
            conn.query(f"SELECT code FROM {STOCK_STABLE} GROUP BY code ORDER BY code;")
        )
    ]


def query_stock_data_by_code(conn, code):
    return pd.DataFrame(
        conn.query(
            f'SELECT {STOCK_VALUES} FROM {STOCK_STABLE} WHERE code = "{code}" ORDER BY ts;'
        ),
        columns=STOCK_VALUES.split(", "),
    )
