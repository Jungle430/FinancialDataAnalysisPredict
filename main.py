import logging
import os
import time

from datetime import datetime

from flask import Flask, app, jsonify
from threadPoolUtil import get_transformer_thread_pool
from apscheduler.schedulers.background import BackgroundScheduler
from transformer import train_model_task, predict
from db import STOCK_VALUES_CN, connection, query_stock_data_by_code
from fileUtil import create_or_clear_directory
from dotenv import load_dotenv

load_dotenv()

td_conn = connection(
    os.getenv("TD_HOST"),
    int(os.getenv("TD_PORT")),  # type: ignore
    os.getenv("TD_USER"),
    os.getenv("TD_PASSWORD"),
    os.getenv("TD_DATABASE"),
)
thread_pool = get_transformer_thread_pool()


scheduler = BackgroundScheduler()
scheduler.add_job(
    func=train_model_task,
    args={td_conn: td_conn, thread_pool: thread_pool},
    trigger="cron",
    hour=5,
    minute=0,
    next_run_time=datetime.now(),
)
scheduler.start()


app = Flask(__name__)

# 禁用 Flask 自身的日志，避免干扰
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


@app.route("/predict/data/<code>", methods=["GET"])  # type: ignore
def get_predict_data(code):
    try:
        predicted_values = predict(
            model_path=f"./model/transformer_predictor_{code}.pth",
            data=query_stock_data_by_code(conn=td_conn, code=code),
            code=code,
        )
        return jsonify(
            {
                "success": True,
                "data": predicted_values[0:].tolist(),
                "err_msg": None,
                "ts": time.time(),
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "data": None,
                "err_msg": str(e),
                "ts": time.time(),
            }
        )


@app.route("/predict/attributes", methods=["GET"])  # type: ignore
def get_predict_attributes():
    return jsonify(
        {
            "success": True,
            "data": STOCK_VALUES_CN,
            "err_msg": None,
            "ts": time.time(),
        }
    )


if __name__ == "__main__":
    create_or_clear_directory("model")
    create_or_clear_directory("runs")
    app.run(debug=False, host="0.0.0.0", port=5000)
