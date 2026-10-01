import os
import json
from application import app
from flask_jwt_extended import create_access_token

def test_jev_endpoint():
    print("Starting API test for Jev Pre-Trade Gate...")
    
    # Configure app for testing
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        with app.app_context():
            # Create a mock JWT token
            access_token = create_access_token(identity="test_user")
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Test the predict endpoint
            ticker = "RELIANCE"
            print(f"Calling POST /api/predict/{ticker}")
            
            response = client.post(
                f'/api/predict/{ticker}',
                headers=headers,
                json={'capital': 100000, 'risk_pct': 2.0}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.get_json()
                print("\n=== Jev Gate Response ===")
                if 'jev_gate' in data:
                    print(json.dumps(data['jev_gate'], indent=2))
                else:
                    print("jev_gate key NOT found in response. (Did the try/except block fail silently?)")
                
                print("\n=== Signal Override Check ===")
                print(f"Final Signal: {data.get('signal')}")
                if 'signal_before_gate' in data:
                    print(f"Original Signal: {data.get('signal_before_gate')}")
                    print(f"Override Reason: {data.get('signal_override_reason')}")
            else:
                print(f"Error Response: {response.get_data(as_text=True)}")

if __name__ == "__main__":
    test_jev_endpoint()
