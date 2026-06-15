import re
import tempfile
import unittest
from pathlib import Path

import web_app


class MimoTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        web_app.app.config.update(
            TESTING=True,
            DATABASE=Path(self.temp_dir.name) / 'test.db',
        )
        with web_app.app.app_context():
            web_app.init_db()
        self.client = web_app.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def csrf_token(self, path):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
        self.assertIsNotNone(match)
        return match.group(1).decode()

    def test_public_pages_and_static_assets(self):
        for path in (
            '/', '/analysis', '/login', '/register',
            '/static/style.css', '/static/mimo_hero.png',
            '/static/mimo_report_woman_red_background.png', '/static/mimo_hero_orange.png',
            '/static/mimo_report_woman_orange.png', '/static/mimo_hero_red.png', '/static/mimo_report_woman_red.png', '/static/favicon.svg',
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()

    def test_mimo_branding_and_analysis_context(self):
        home = self.client.get('/')
        analysis = self.client.get('/analysis')
        self.assertIn(b'MIMO', home.data)
        self.assertIn(b'mimo_hero_red.png', home.data)
        self.assertNotIn(b'mimo_hero_orange.png', home.data)
        self.assertIn(b'mimo_report_woman_red_background.png', analysis.data)
        self.assertNotIn(b'mimo_report_woman_orange.png', analysis.data)
        self.assertIn(b'mimo-symbol', home.data)
        self.assertIn(b'logo-wordmark', home.data)
        self.assertIn('美'.encode(), home.data)
        self.assertIn(b'data-mobile-back', home.data)
        self.assertIn(b'dock-icon', home.data)
        self.assertIn(b'responsiveViewQuery', home.data)
        self.assertNotIn(b'mimo-view', home.data)
        self.assertIn(b'data-view-mode="desktop"', home.data)
        self.assertNotIn(b'data-view-mode="auto"', home.data)
        self.assertIn(b'data-view-mode="mobile"', home.data)
        self.assertIn(b'width=1280', home.data)
        self.assertIn(b'mobile-version-frame', home.data)
        self.assertIn('요즘 많이 찾는'.encode(), analysis.data)
        self.assertIn('제품 전성분을 직접 비교한 결과는 아니므로'.encode(), analysis.data)
        self.assertEqual(analysis.data.count(b'class="trend-rail-list"'), 1)
        with web_app.app.app_context():
            self.assertEqual(len(web_app.get_trending_terms()), 5)

    def test_expert_recommendation_and_invalid_mode(self):
        response = self.client.post('/recommend', json={
            'keyword': '피부가 건조하고 각질이 생겨요',
            'gender': '여성',
            'age': '20대',
            'skin_type': '건성',
            'mode': 'expert',
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['mode'], 'expert')
        basis = data['analysis_basis']
        self.assertEqual(basis['dataset_count'], len(web_app.df_skin))
        self.assertEqual(basis['method'], 'tfidf_profile')
        self.assertTrue(basis['extracted_terms'])
        self.assertIsNotNone(basis['text_similarity_percent'])
        self.assertIn('matched_question', basis)
        self.assertIn('Recommended Ingredients', basis['source_fields'])
        selected = web_app.select_expert_row(
            '피부가 건조하고 각질이 생겨요',
            web_app.preprocess_input('피부가 건조하고 각질이 생겨요'),
            '여성', '20대', '건성',
        )
        self.assertEqual(
            (selected['Gender'], selected['Age'], selected['Skin Type']),
            ('여성', '20대', '건성'),
        )

        response = self.client.post('/recommend', json={
            'keyword': '건조함',
            'mode': 'unsupported',
        })
        self.assertEqual(response.status_code, 400)

    def test_expert_selection_matches_v4_rebuild_formula(self):
        cases = (
            ('피부가 건조하고 각질이 생겨요', '여성', '20대', '건성'),
            ('붉고 민감하며 화장이 들떠요', '여성', '30대', '복합성'),
            ('모공과 피지가 많고 번들거려요', '남성', '30대', '지성'),
        )
        for text, gender, age, skin_type in cases:
            with self.subTest(text=text):
                cleaned = web_app.preprocess_input(text)
                similarities = web_app.linear_kernel(
                    web_app.tfidf.transform([cleaned]),
                    web_app.tfidf_matrix,
                )[0]
                profile_bonus = web_app.np.zeros(len(web_app.df_skin))
                profile_bonus += web_app.df_skin['Gender'].eq(gender).to_numpy() * 0.05
                profile_bonus += web_app.df_skin['Age'].eq(age).to_numpy() * 0.05
                profile_bonus += web_app.df_skin['Skin Type'].eq(skin_type).to_numpy() * 0.05
                expected_index = int((similarities + profile_bonus).argmax())
                selected = web_app.select_expert_row(
                    text, cleaned, gender, age, skin_type,
                )
                self.assertEqual(selected.name, expected_index)

    def test_rebuild_resources_and_preprocessing_are_aligned(self):
        self.assertEqual(len(web_app.df_skin), web_app.tfidf_matrix.shape[0])
        self.assertEqual(
            web_app.tfidf_matrix.shape[1],
            len(web_app.tfidf.get_feature_names_out()),
        )
        sample_indices = (
            0,
            len(web_app.df_skin) // 4,
            len(web_app.df_skin) // 2,
            len(web_app.df_skin) * 3 // 4,
            len(web_app.df_skin) - 1,
        )
        for index in sample_indices:
            row = web_app.df_skin.iloc[index]
            self.assertEqual(
                web_app.preprocess_input(row['User Question']),
                row['cleaned_question'],
            )
            sample_vec = web_app.tfidf.transform([row['cleaned_question']])
            similarities = web_app.linear_kernel(
                sample_vec, web_app.tfidf_matrix,
            )[0]
            self.assertEqual(int(similarities.argmax()), index)
            self.assertGreater(float(similarities[index]), 0.999)

    def test_review_recommendation_uses_skincare_rows(self):
        response = self.client.post("/recommend", json={
            "keyword": "피부 트러블과 여드름이 반복돼요",
            "mode": "review",
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["mode"], "review")
        self.assertTrue(data["results"])
        self.assertTrue(all(
            result["category"] in web_app.SKINCARE_CATEGORIES
            for result in data["results"]
        ))

    def test_empty_recommendation_is_rejected(self):
        response = self.client.post('/recommend', json={})
        self.assertEqual(response.status_code, 400)

    def test_registration_rejects_tampered_profile_value(self):
        response = self.client.post('/register', data={
            'csrf_token': self.csrf_token('/register'),
            'name': '테스트',
            'username': 'test_user',
            'email': 'test@example.com',
            'password': 'Password1',
            'password_confirm': 'Password1',
            'gender': '성별 선택',
            'age': '20대',
            'skin_type': '건성',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('올바른 성별 값을 선택해 주세요.'.encode(), response.data)

    def test_admin_login_and_dashboard(self):
        response = self.client.post('/login', data={
            'csrf_token': self.csrf_token('/login'),
            'username': 'admin',
            'password': 'Admin1234!',
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('사용자 관리'.encode(), response.data)


if __name__ == '__main__':
    unittest.main()
