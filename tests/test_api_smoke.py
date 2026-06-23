from fastapi.testclient import TestClient

from api.main import app, chat_job_runner


def test_api_health_and_session_crud():
    with TestClient(app) as client:
        health = client.get('/api/v1/health')
        assert health.status_code == 200
        assert health.json().get('ok') is True

        created = client.post('/api/v1/sessions', json={'title': 'pytest-session'})
        assert created.status_code == 200
        session_id = created.json()['id']

        detail = client.get(f'/api/v1/sessions/{session_id}')
        assert detail.status_code == 200
        assert detail.json()['session']['id'] == session_id

        cleared = client.delete(f'/api/v1/sessions/{session_id}/messages')
        assert cleared.status_code == 200

        deleted = client.delete(f'/api/v1/sessions/{session_id}')
        assert deleted.status_code == 200


def test_api_chat_job_create_and_active_list(monkeypatch):
    monkeypatch.setattr(chat_job_runner, "submit", lambda job_id: None)

    with TestClient(app) as client:
        created = client.post('/api/v1/sessions', json={'title': 'pytest-job-session'})
        assert created.status_code == 200
        session_id = created.json()['id']

        job_resp = client.post(
            '/api/v1/chat/jobs',
            json={'session_id': session_id, 'message': '扫地机器人回充失败怎么办？'},
        )
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job['session_id'] == session_id
        assert job['status'] == 'queued'
        assert job['prompt'] == '扫地机器人回充失败怎么办？'

        active_resp = client.get('/api/v1/chat/jobs/active')
        assert active_resp.status_code == 200
        active_ids = {item['id'] for item in active_resp.json()}
        assert job['id'] in active_ids

        detail = client.get(f"/api/v1/chat/jobs/{job['id']}")
        assert detail.status_code == 200
        assert detail.json()['id'] == job['id']

        messages = client.get(f'/api/v1/sessions/{session_id}/messages')
        assert messages.status_code == 200
        assert messages.json()[0]['role'] == 'user'

        deleted = client.delete(f'/api/v1/sessions/{session_id}')
        assert deleted.status_code == 200
