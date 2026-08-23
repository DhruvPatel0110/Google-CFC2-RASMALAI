import unittest
from fastapi.testclient import TestClient
from src.app.api import app

class TestAppAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], "HEALTHY")

    def test_facilities_endpoint(self):
        response = self.client.get("/api/v1/network/facilities")
        self.assertEqual(response.status_code, 200)
        facilities = response.json()
        self.assertEqual(len(facilities), 15)

    def test_medicine_stockouts_endpoint(self):
        response = self.client.get("/api/v1/medicine/stockouts?date=2025-08-31")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_bed_occupancy_endpoint(self):
        response = self.client.get("/api/v1/beds/occupancy?date=2025-08-31")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 15)

    def test_staff_attendance_endpoint(self):
        response = self.client.get("/api/v1/staff/attendance?date=2025-08-31")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 15)

    def test_redistribution_plan_endpoint(self):
        response = self.client.get("/api/v1/redistribution/plan?date=2025-08-31")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("impact_metrics", data)

    def test_federation_model_card_endpoint(self):
        response = self.client.get("/api/v1/federation/model-card")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_title", data)

if __name__ == '__main__':
    unittest.main()
