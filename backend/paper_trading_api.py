from flask import Blueprint, jsonify, request, g
from flask_jwt_extended import jwt_required
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

def create_paper_trading_blueprint(order_manager, broker):
    """
    Creates a Flask Blueprint for Paper Trading and Order Management.
    """
    bp = Blueprint('paper_trading', __name__, url_prefix='/api/paper')

    @bp.route('/funds', methods=['GET'])
    @jwt_required()
    def get_funds():
        """Get paper trading capital and margin usage."""
        try:
            return jsonify(broker.get_funds())
        except Exception as e:
            logger.error(f"Error fetching funds: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @bp.route('/positions', methods=['GET'])
    @jwt_required()
    def get_positions():
        """Get current open positions and live PnL."""
        try:
            positions = broker.get_positions()
            return jsonify([asdict(p) for p in positions])
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @bp.route('/orders', methods=['GET'])
    @jwt_required()
    def get_orders():
        """Get today's orders."""
        try:
            orders = broker.get_order_history()
            return jsonify([asdict(o) for o in orders])
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @bp.route('/orders/<order_id>', methods=['DELETE'])
    @jwt_required()
    def cancel_order(order_id):
        """Cancel a pending order."""
        try:
            success = broker.cancel_order(order_id)
            if success:
                return jsonify({"message": f"Order {order_id} cancelled successfully"})
            return jsonify({"error": f"Order {order_id} not found or not pending"}), 404
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @bp.route('/audit', methods=['GET'])
    @jwt_required()
    def get_audit():
        """Get safety layer audit trail (rejections, confirmations, execution logs)."""
        limit = request.args.get('limit', 50, type=int)
        try:
            audit = order_manager.get_audit_trail(limit=limit)
            return jsonify(audit)
        except Exception as e:
            logger.error(f"Error fetching audit trail: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @bp.route('/stats', methods=['GET'])
    @jwt_required()
    def get_stats():
        """Get OrderManager daily safety stats (PnL limits, order counts, kill switch status)."""
        try:
            return jsonify(order_manager.get_daily_stats())
        except Exception as e:
            logger.error(f"Error fetching daily stats: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @bp.route('/kill', methods=['POST'])
    @jwt_required()
    def kill_switch():
        """Emergency stop all trading."""
        try:
            return jsonify(order_manager.kill_switch())
        except Exception as e:
            logger.error(f"Error activating kill switch: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @bp.route('/resume', methods=['POST'])
    @jwt_required()
    def resume_trading():
        """Resume trading after kill switch."""
        try:
            return jsonify(order_manager.resume_trading())
        except Exception as e:
            logger.error(f"Error resuming trading: {e}")
            return jsonify({"error": "Internal server error"}), 500

    return bp
