import requests
import json
from typing import Dict, Any, Optional
from config import Config


class ZohoBooksClient:
    def __init__(self):
        self.client_id = Config.ZOHO_CLIENT_ID
        self.client_secret = Config.ZOHO_CLIENT_SECRET
        self.refresh_token = Config.ZOHO_REFRESH_TOKEN
        self.organization_id = Config.ZOHO_ORGANIZATION_ID
        self.api_domain = Config.ZOHO_API_DOMAIN
        self.access_token = None
    
    def _get_access_token(self) -> str:
        if self.access_token:
            return self.access_token
        
        url = "https://accounts.zoho.com/oauth/v2/token"
        data = {
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token"
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        token_data = response.json()
        self.access_token = token_data.get("access_token")
        return self.access_token
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Zoho-oauthtoken {self._get_access_token()}",
            "Content-Type": "application/json"
        }
    
    def create_expense(self,
                       vendor_name: str,
                       amount: float,
                       currency: str = "INR",
                       date: Optional[str] = None,
                       reference_number: Optional[str] = None,
                       description: Optional[str] = None) -> Dict[str, Any]:
        
        if not self.refresh_token or self.refresh_token.startswith("1000.x"):
            return {"error": "Zoho credentials not configured", "status": "skipped"}
        
        url = f"{self.api_domain}/expenses"
        
        from datetime import datetime
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        payload = {
            "date": date,
            "amount": amount,
            "currency_id": currency,
            "vendor_name": vendor_name,
            "reference_number": reference_number or "",
            "description": description or f"Bill from {vendor_name}"
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                params={"organization_id": self.organization_id},
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
