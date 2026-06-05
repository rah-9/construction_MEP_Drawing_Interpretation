import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from schemas.response_schema import InterpretationResponse
from schemas.document_schema import DocumentRepository, PageEntry, PageDimension, OCRElement, LayoutRegion
from core.layout_extractor import LayoutExtractor
from core.validator import Validator
from core.confidence_engine import ConfidenceEngine
from core.evidence_store import EvidenceStore
from core.question_analyzer import QuestionAnalyzer
from core.context_retriever import ContextRetriever
from core.model_verifier import ModelVerifier
from core.counting_handler import CountingHandler
from models.nvidia_client import NVIDIAClient

class TestLayoutExtractor(unittest.TestCase):
    def test_default_layout_regions(self):
        # Mock fitz.Page
        mock_page = MagicMock()
        mock_page.rect.width = 1000
        mock_page.rect.height = 800
        # Mock search_for to return empty (simulates scanned drawing or no keywords)
        mock_page.search_for.return_value = []
        
        regions = LayoutExtractor.extract_regions(mock_page)
        
        # Check that we got 4 distinct layout regions
        self.assertEqual(len(regions), 4)
        
        region_types = {r.region_type for r in regions}
        self.assertIn("title_block", region_types)
        self.assertIn("legend", region_types)
        self.assertIn("notes", region_types)
        self.assertIn("drawing_area", region_types)

        # Check bounds: drawing area should be mostly on the left
        drawing_region = next(r for r in regions if r.region_type == "drawing_area")
        ymin, xmin, ymax, xmax = drawing_region.bbox
        self.assertLess(xmin, 0.5)
        self.assertLess(xmax, 0.8)


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.validator = Validator()
        self.mock_page_entry = PageEntry(
            page_number=1,
            dimensions=PageDimension(width=1000, height=800),
            image_path="dummy.png",
            preview_path="dummy_preview.png",
            page_hash="dummyhash",
            ocr_elements=[
                OCRElement(text="Control", bbox=[0.5, 0.5, 0.6, 0.6], confidence=1.0, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="Point", bbox=[0.5, 0.6, 0.6, 0.7], confidence=1.0, page_number=1, evidence_type="ocr_text")
            ]
        )
        self.mock_repo = DocumentRepository(
            document_name="test.pdf",
            file_hash="dummyhash",
            pages=[self.mock_page_entry]
        )
        self.evidence_store = EvidenceStore(self.mock_repo)

    def test_validation_page_mismatch(self):
        # Page mismatch
        response = InterpretationResponse(
            answer="CP",
            confidence=0.9,
            page_number=2,  # mismatch with page_entry page_number=1
            bbox=[[0.1, 0.1, 0.2, 0.2]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )
        is_valid, msg, matches = self.validator.validate(response, self.mock_page_entry, self.evidence_store)
        self.assertFalse(is_valid)
        self.assertIn("page number", msg.lower())

    def test_validation_missing_bbox(self):
        # Missing bbox when evidence declared found
        response = InterpretationResponse(
            answer="FCU-1",
            confidence=0.95,
            page_number=1,
            bbox=[],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )
        is_valid, msg, matches = self.validator.validate(response, self.mock_page_entry, self.evidence_store)
        self.assertFalse(is_valid)
        self.assertIn("bbox", msg.lower())
        self.assertIn("empty", msg.lower())

    def test_validation_out_of_bounds_bbox(self):
        # Bbox out of range [0.0, 1.0]
        response = InterpretationResponse(
            answer="FCU-1",
            confidence=0.95,
            page_number=1,
            bbox=[[0.1, -0.5, 0.2, 1.2]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )
        is_valid, msg, matches = self.validator.validate(response, self.mock_page_entry, self.evidence_store)
        self.assertFalse(is_valid)
        self.assertIn("out of page boundaries", msg.lower())

    def test_ocr_cross_verification_warning(self):
        # Keyword "VAV" is not in OCR, should produce warning/missing flag in message
        response = InterpretationResponse(
            answer="VAV is installed",
            confidence=0.95,
            page_number=1,
            bbox=[[0.1, 0.1, 0.2, 0.2]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )
        is_valid, msg, matches = self.validator.validate(response, self.mock_page_entry, self.evidence_store)
        self.assertTrue(is_valid)
        self.assertIn("OCR Cross-Verification Warning", msg)


class TestConfidenceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ConfidenceEngine()

    def test_high_confidence(self):
        # High confidence should be calculated when OCR match, agreement, bbox, and density are high.
        result = self.engine.calculate_final_confidence(
            model_confidence=0.95,
            ocr_match_score=1.0,
            agreement_score=1.0,
            has_bbox=True,
            ocr_elements_count=5
        )
        self.assertGreaterEqual(result["final_score"], 0.90)
        self.assertEqual(result["confidence_label"], "High")

    def test_low_confidence(self):
        # Low confidence/human review should trigger when elements are missing.
        result = self.engine.calculate_final_confidence(
            model_confidence=0.30,
            ocr_match_score=0.1,
            agreement_score=0.2,
            has_bbox=False,
            ocr_elements_count=0
        )
        self.assertLess(result["final_score"], 0.50)
        self.assertEqual(result["confidence_label"], "Human Review Required")


class TestEvidenceStore(unittest.TestCase):
    def setUp(self):
        self.mock_page_entry = PageEntry(
            page_number=1,
            dimensions=PageDimension(width=1000, height=800),
            image_path="dummy.png",
            preview_path="dummy_preview.png",
            page_hash="dummyhash",
            layout_regions=[
                LayoutRegion(region_type="legend", bbox=[0.0, 0.8, 0.5, 1.0], bbox_pts=[0.0, 640.0, 500.0, 800.0]),
                LayoutRegion(region_type="notes", bbox=[0.5, 0.8, 1.0, 1.0], bbox_pts=[500.0, 640.0, 1000.0, 800.0])
            ],
            ocr_elements=[
                OCRElement(text="Control", bbox=[0.5, 0.5, 0.6, 0.6], confidence=1.0, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="Point", bbox=[0.5, 0.6, 0.6, 0.7], confidence=1.0, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="FCU-1", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.98, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="LegendItem", bbox=[0.1, 0.85, 0.2, 0.95], confidence=0.99, page_number=1, evidence_type="legend"),
                OCRElement(text="NoteItem", bbox=[0.6, 0.85, 0.7, 0.95], confidence=0.99, page_number=1, evidence_type="notes")
            ]
        )
        self.mock_repo = DocumentRepository(
            document_name="test.pdf",
            file_hash="dummyhash",
            pages=[self.mock_page_entry]
        )
        self.store = EvidenceStore(self.mock_repo)

    def test_search_single_word(self):
        results = self.store.search_text("fcu-1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "FCU-1")

    def test_search_phrase(self):
        # Search multi-word phrase in reading order
        results = self.store.search_text("Control Point")
        self.assertEqual(len(results), 2)
        texts = {r.text for r in results}
        self.assertIn("Control", texts)
        self.assertIn("Point", texts)

    def test_search_by_type(self):
        results = self.store.search_by_type("legend")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "LegendItem")

    def test_search_by_page(self):
        results = self.store.search_by_page(1)
        self.assertEqual(len(results), 5)

    def test_search_by_region(self):
        # LegendItem is inside Legend region [0.0, 0.8, 0.5, 1.0]
        results = self.store.search_by_region("legend", page_num=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "LegendItem")

    def test_search_fuzzy(self):
        # Fuzzy match "Control"
        results = self.store.search_fuzzy("Contrl", threshold=80, page_num=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Control")

    def test_search_nearest_text(self):
        # "Point" is adjacent to "Control", whereas "FCU-1" is far away
        results = self.store.search_nearest_text("Control", page_num=1, limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Point")


class TestQuestionAnalyzer(unittest.TestCase):
    def test_analyze_question_counting(self):
        analysis = QuestionAnalyzer.analyze_question("How many FCUs are on this page?")
        self.assertEqual(analysis["question_type"], "counting")
        self.assertIn("legend", analysis["retrieval_strategy"].lower())

    def test_analyze_question_legend(self):
        analysis = QuestionAnalyzer.analyze_question("What does the symbol VAV stand for?")
        self.assertEqual(analysis["question_type"], "abbreviation")

    def test_analyze_question_notes(self):
        analysis = QuestionAnalyzer.analyze_question("Read note 5 for installation details.")
        self.assertEqual(analysis["question_type"], "notes")

    def test_analyze_question_conflict(self):
        analysis = QuestionAnalyzer.analyze_question("Is there a clash between duct and beam?")
        self.assertEqual(analysis["question_type"], "conflict")

    def test_analyze_question_location(self):
        analysis = QuestionAnalyzer.analyze_question("Where is the FCU located?")
        self.assertEqual(analysis["question_type"], "location")

    def test_analyze_question_equipment(self):
        analysis = QuestionAnalyzer.analyze_question("What is the capacity of FCU-1?")
        self.assertEqual(analysis["question_type"], "equipment")

    def test_analyze_question_general(self):
        analysis = QuestionAnalyzer.analyze_question("Who drew this sheet?")
        self.assertEqual(analysis["question_type"], "general")


class TestContextRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_page_entry = PageEntry(
            page_number=1,
            dimensions=PageDimension(width=1000, height=800),
            image_path="dummy.png",
            preview_path="dummy_preview.png",
            page_hash="dummyhash",
            layout_regions=[
                LayoutRegion(region_type="legend", bbox=[0.0, 0.8, 0.5, 1.0], bbox_pts=[0.0, 640.0, 500.0, 800.0]),
                LayoutRegion(region_type="notes", bbox=[0.5, 0.8, 1.0, 1.0], bbox_pts=[500.0, 640.0, 1000.0, 800.0])
            ],
            ocr_elements=[
                OCRElement(text="Control", bbox=[0.5, 0.5, 0.6, 0.6], confidence=1.0, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="Point", bbox=[0.5, 0.6, 0.6, 0.7], confidence=1.0, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="LegendItem", bbox=[0.1, 0.85, 0.2, 0.95], confidence=0.99, page_number=1, evidence_type="legend"),
                OCRElement(text="NoteItem", bbox=[0.6, 0.85, 0.7, 0.95], confidence=0.99, page_number=1, evidence_type="notes")
            ]
        )
        self.mock_repo = DocumentRepository(
            document_name="test.pdf",
            file_hash="dummyhash",
            pages=[self.mock_page_entry]
        )
        self.store = EvidenceStore(self.mock_repo)

    def test_format_elements(self):
        elements = [self.mock_page_entry.ocr_elements[0]]
        formatted = ContextRetriever.format_elements(elements)
        self.assertIn("Control", formatted)
        self.assertIn("at y: 0.50, x: 0.50", formatted)

    def test_retrieve_context_legend(self):
        context = ContextRetriever.retrieve_context(self.store, 1, "What does LegendItem mean?", "legend")
        self.assertIn("LegendItem", context)


class TestModelVerifier(unittest.TestCase):
    def test_verifier_agreement(self):
        resp1 = InterpretationResponse(
            answer="FCU-1 has 500 CFM capacity",
            confidence=0.95,
            page_number=1,
            bbox=[[0.1, 0.1, 0.2, 0.2]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )
        resp2 = InterpretationResponse(
            answer="FCU-1 has 500 CFM capacity",
            confidence=0.90,
            page_number=1,
            bbox=[[0.1, 0.1, 0.22, 0.22]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )

        results = ModelVerifier.verify(resp1, resp2)
        # Bbox overlap (IoU) should be high and text is identical
        self.assertGreaterEqual(results["agreement_score"], 0.80)
        self.assertIn("high agreement", results["disagreement_reason"].lower())

    def test_verifier_disagreement(self):
        resp1 = InterpretationResponse(
            answer="FCU-1 has 500 CFM",
            confidence=0.95,
            page_number=1,
            bbox=[[0.1, 0.1, 0.2, 0.2]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )
        resp2 = InterpretationResponse(
            answer="VAV-3 is at 300 CFM",
            confidence=0.90,
            page_number=1,
            bbox=[[0.7, 0.7, 0.8, 0.8]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Test"
        )

        results = ModelVerifier.verify(resp1, resp2)
        self.assertLess(results["agreement_score"], 0.50)


class TestCountingHandler(unittest.TestCase):
    def setUp(self):
        self.mock_page_entry = PageEntry(
            page_number=1,
            dimensions=PageDimension(width=1000, height=800),
            image_path="dummy.png",
            preview_path="dummy_preview.png",
            page_hash="dummyhash",
            ocr_elements=[
                OCRElement(text="FD", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="FD", bbox=[0.3, 0.3, 0.4, 0.4], confidence=0.9, page_number=1, evidence_type="ocr_text")
            ]
        )
        self.mock_repo = DocumentRepository(
            document_name="test.pdf",
            file_hash="dummyhash",
            pages=[self.mock_page_entry]
        )
        self.store = EvidenceStore(self.mock_repo)

    def test_extract_target_object(self):
        target = CountingHandler.extract_target_object("How many fire dampers are present?")
        self.assertEqual(target, "fd")  # abbreviation mapping in CountingHandler

        target2 = CountingHandler.extract_target_object("Count the number of VAV boxes.")
        self.assertEqual(target2, "vav")

    def test_verify_counting_evidence(self):
        # 2 fire dampers in OCR, model claims 2
        results = CountingHandler.verify_counting_evidence("There are 2 fire dampers", "fd", 1, self.store)
        self.assertEqual(results["model_count"], 2)
        self.assertEqual(results["ocr_count"], 2)
        self.assertEqual(results["ocr_support"], 1.0)

    def test_adjust_counting_confidence(self):
        # High agreement, high OCR support
        adj = CountingHandler.adjust_counting_confidence(0.95, 1.0, 0.90)
        self.assertEqual(adj, 0.95)

        # Low agreement or OCR support should cap at 0.79
        adj2 = CountingHandler.adjust_counting_confidence(0.95, 0.5, 0.90)
        self.assertEqual(adj2, 0.79)


class TestNVIDIAClient(unittest.TestCase):
    @patch('config.NVIDIA_API_KEY', 'test_key')
    @patch('config.NVIDIA_ENDPOINT', 'https://test.api.nvidia.com/v1/chat/completions')
    @patch('config.NVIDIA_PRIMARY_MODEL', 'moonshotai/kimi-k2.6')
    @patch('config.NVIDIA_PAYLOAD_FORMAT', 'array')
    def test_nvidia_client_init(self):
        client = NVIDIAClient()
        self.assertTrue(client.client_active)
        self.assertEqual(client.endpoint, 'https://test.api.nvidia.com/v1/chat/completions')
        self.assertEqual(client.model, 'moonshotai/kimi-k2.6')
        self.assertEqual(client.payload_format, 'array')

    @patch('config.NVIDIA_API_KEY', 'test_key')
    @patch('requests.post')
    def test_nvidia_client_health_check_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client = NVIDIAClient()
        is_healthy, msg = client.health_check()
        self.assertTrue(is_healthy)
        self.assertEqual(msg, "Endpoint accessible and active.")

    @patch('config.NVIDIA_API_KEY', 'test_key')
    @patch('requests.post')
    def test_nvidia_client_health_check_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = NVIDIAClient()
        is_healthy, msg = client.health_check()
        self.assertFalse(is_healthy)
        self.assertIn("HTTP Error 500", msg)

    @patch('config.NVIDIA_API_KEY', 'test_key')
    @patch('config.NVIDIA_ENDPOINT', 'https://test.api.nvidia.com/v1/chat/completions')
    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_bytes')
    @patch('pathlib.Path.exists', return_value=True)
    def test_nvidia_client_query_200_json_success(self, mock_exists, mock_file, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Success raw 200"
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "Room 101", "confidence": 0.95, "page_number": 1, "bbox": [[0.1, 0.1, 0.2, 0.2]], "evidence_found": true, "review_required": false, "reasoning_summary": "Test"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        client = NVIDIAClient()
        client.payload_format = "array"
        
        response = client.query(
            image_path=Path("dummy_page.png"),
            retrieved_context="some context",
            question="Where is the unit?",
            page_number=1,
            prompt_template="{context} {question} {page_number}"
        )

        self.assertEqual(response.answer, "Room 101")
        self.assertEqual(response.confidence, 0.95)
        # Check that post was called with array format content
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        user_content = payload["messages"][1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")

    @patch('config.NVIDIA_API_KEY', 'test_key')
    @patch('config.NVIDIA_ENDPOINT', 'https://test.api.nvidia.com/v1/chat/completions')
    @patch('requests.get')
    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_bytes')
    @patch('pathlib.Path.exists', return_value=True)
    def test_nvidia_client_query_202_polling(self, mock_exists, mock_file, mock_post, mock_get):
        # 1. Mock initial 202 response
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 202
        mock_post_resp.text = "Accepted"
        mock_post_resp.json.return_value = {
            "requestId": "test-uuid-1234",
            "message": "Result is pending."
        }
        mock_post.return_value = mock_post_resp

        # 2. Mock polling responses: first returns 202, second returns 200 with result
        mock_get_resp1 = MagicMock()
        mock_get_resp1.status_code = 202
        mock_get_resp1.text = "Pending"

        mock_get_resp2 = MagicMock()
        mock_get_resp2.status_code = 200
        mock_get_resp2.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "Grounded Value", "confidence": 0.90, "page_number": 1, "bbox": [], "evidence_found": true, "review_required": false, "reasoning_summary": "Polled text"}'
                    }
                }
            ]
        }
        mock_get_resp2.text = "Success JSON"

        mock_get.side_effect = [mock_get_resp1, mock_get_resp2]

        client = NVIDIAClient()
        response = client.query(
            image_path=Path("dummy_page.png"),
            retrieved_context="some context",
            question="Question?",
            page_number=1,
            prompt_template="{context} {question} {page_number}"
        )

        self.assertEqual(response.answer, "Grounded Value")
        self.assertEqual(response.confidence, 0.90)
        self.assertEqual(mock_get.call_count, 2)
        # Verify status endpoint is correct
        mock_get.assert_called_with(
            "https://test.api.nvidia.com/v1/status/test-uuid-1234",
            headers={"Authorization": "Bearer test_key", "accept": "application/json"},
            timeout=10
        )

    @patch('config.NVIDIA_API_KEY', 'test_key')
    @patch('config.NVIDIA_ENDPOINT', 'https://test.api.nvidia.com/v1/chat/completions')
    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_bytes')
    @patch('pathlib.Path.exists', return_value=True)
    def test_nvidia_client_plain_text_normalization(self, mock_exists, mock_file, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Raw plain text answer"
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This drawing has a drawing number of M-101."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        client = NVIDIAClient()
        response = client.query(
            image_path=Path("dummy_page.png"),
            retrieved_context="some context",
            question="Question?",
            page_number=1,
            prompt_template="{context} {question} {page_number}"
        )

        # Should normalize it with safe defaults
        # Should normalize it with safe defaults
        self.assertEqual(response.answer, "This drawing has a drawing number of M-101.")
        self.assertEqual(response.confidence, 0.5)
        self.assertTrue(response.review_required)
        self.assertEqual(response.page_number, 1)
        self.assertEqual(response.bbox, [])


class TestV3Engines(unittest.TestCase):
    def setUp(self):
        # Setup mock document repo
        self.mock_page_1 = PageEntry(
            page_number=1,
            dimensions=PageDimension(width=1000, height=800),
            image_path="dummy1.png",
            preview_path="dummy1_preview.png",
            page_hash="dummyhash1",
            layout_regions=[
                LayoutRegion(region_type="notes", bbox=[0.5, 0.8, 1.0, 1.0], bbox_pts=[500.0, 640.0, 1000.0, 800.0])
            ],
            ocr_elements=[
                OCRElement(text="GENERAL", bbox=[0.51, 0.81, 0.55, 0.85], confidence=0.99, page_number=1, evidence_type="notes"),
                OCRElement(text="NOTES", bbox=[0.51, 0.86, 0.55, 0.90], confidence=0.99, page_number=1, evidence_type="notes"),
                OCRElement(text="Section", bbox=[0.6, 0.1, 0.65, 0.2], confidence=0.95, page_number=1, evidence_type="ocr_text"),
                OCRElement(text="15010", bbox=[0.6, 0.21, 0.65, 0.3], confidence=0.95, page_number=1, evidence_type="ocr_text")
            ]
        )
        self.mock_page_2 = PageEntry(
            page_number=2,
            dimensions=PageDimension(width=1000, height=800),
            image_path="dummy2.png",
            preview_path="dummy2_preview.png",
            page_hash="dummyhash2",
            layout_regions=[
                LayoutRegion(region_type="legend", bbox=[0.0, 0.8, 0.5, 1.0], bbox_pts=[0.0, 640.0, 500.0, 800.0])
            ],
            ocr_elements=[
                OCRElement(text="CD-1", bbox=[0.1, 0.81, 0.2, 0.89], confidence=0.99, page_number=2, evidence_type="legend"),
                OCRElement(text="CEILING", bbox=[0.1, 0.9, 0.2, 0.95], confidence=0.99, page_number=2, evidence_type="legend"),
                OCRElement(text="DIFFUSER", bbox=[0.1, 0.96, 0.2, 0.99], confidence=0.99, page_number=2, evidence_type="legend"),
                OCRElement(text="FCU-1", bbox=[0.3, 0.3, 0.35, 0.4], confidence=0.98, page_number=2, evidence_type="ocr_text")
            ]
        )
        self.repo = DocumentRepository(
            document_name="doc.pdf",
            file_hash="dummyhash",
            pages=[self.mock_page_1, self.mock_page_2]
        )
        
        # Instantiate engines
        from core.document_intelligence import DocumentIntelligence
        from core.confidence_engine import ConfidenceEngine
        self.repo.doc_intelligence = DocumentIntelligence.analyze(self.repo)
        self.store = EvidenceStore(self.repo)
        self.engine = ConfidenceEngine()

    def test_document_intelligence_pages_classification(self):
        doc_intel = self.repo.doc_intelligence
        self.assertIn(1, doc_intel["notes_pages"])
        self.assertIn(2, doc_intel["legend_pages"])
        self.assertIn("CD-1", doc_intel["symbols"])
        self.assertIn("FCU-1", doc_intel["equipment_tags"])

    def test_page_ranker_legend_query(self):
        from core.page_ranker import PageRanker
        rankings = PageRanker.rank_pages("What does CD-1 mean?", "legend", self.store, self.repo.doc_intelligence)
        # Page 2 (legend page and contains CD-1) should rank higher than Page 1
        self.assertEqual(rankings[0]["page"], 2)
        self.assertGreater(rankings[0]["score"], rankings[1]["score"])

    def test_context_builder(self):
        from core.context_builder import ContextBuilder
        context = ContextBuilder.build_context(self.store, [2], "What is CD-1?", "legend", self.repo.doc_intelligence)
        self.assertIn("CD-1", context)
        self.assertIn("CEILING", context)

    def test_session_memory(self):
        from core.session_memory import SessionMemory
        memory = SessionMemory()
        
        # Test symbol resolution
        memory.add_qa("What does CD-1 mean?", "CD-1 means Ceiling Diffuser.")
        ctx = memory.get_injected_memory_context("Show CD-1 airflow.")
        self.assertIn("Ceiling Diffuser", ctx)
        
        # Test QA history matching
        memory.add_qa("Airflow details for FCU-1?", "The airflow is 400 CFM.")
        ctx_qa = memory.get_injected_memory_context("What airflow is FCU-1?")
        self.assertIn("airflow is 400 CFM", ctx_qa)
        
        # Test clear
        memory.clear()
        self.assertEqual(memory.get_injected_memory_context("What is CD-1?"), "")

    def test_counting_engine(self):
        from core.counting_engine import CountingEngine
        
        p_resp = InterpretationResponse(
            answer="There are 3 diffusers.",
            confidence=0.9,
            page_number=2,
            bbox=[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Counted visually."
        )
        
        # Verifier agrees
        v_resp = InterpretationResponse(
            answer="I counted 3 diffusers.",
            confidence=0.85,
            page_number=2,
            bbox=[[0.1, 0.1, 0.2, 0.2]],
            evidence_found=True,
            review_required=False,
            reasoning_summary="Agreed."
        )
        
        # Target has 1 occurrence in OCR page 2 ("CD-1")
        audit = CountingEngine.audit_count("How many ceiling diffusers?", p_resp, v_resp, self.store, [2])
        self.assertEqual(audit["visual_count"], 3)
        self.assertEqual(audit["verifier_count"], 3)
        self.assertEqual(audit["verification_support"], 1.0)
        self.assertLess(audit["ocr_support"], 1.0) # mismatch between 3 and 1

    def test_intent_aware_confidence_engine(self):
        # Verify custom weights are applied for legend intent
        result = self.engine.calculate_final_confidence(
            model_confidence=0.90,
            ocr_match_score=0.80,
            agreement_score=1.00,
            has_bbox=True,
            ocr_elements_count=3,
            intent="legend"
        )
        # Score calculation: 0.40 * 0.80 + 0.30 * 1.00 + 0.20 * 0.90 + 0.10 * 1.0 = 0.32 + 0.30 + 0.18 + 0.1 = 0.90
        self.assertEqual(result["final_score"], 0.90)
        self.assertEqual(result["confidence_label"], "High")

    def test_confidence_consistency_capping(self):
        # 1. review_required=True should cap at 0.69
        result = self.engine.calculate_final_confidence(
            model_confidence=0.95,
            ocr_match_score=1.0,
            agreement_score=1.0,
            has_bbox=True,
            ocr_elements_count=5,
            review_required=True
        )
        self.assertLessEqual(result["final_score"], 0.69)
        self.assertEqual(result["confidence_label"], "Low")

        # 2. evidence_found=False should cap at 0.59
        result2 = self.engine.calculate_final_confidence(
            model_confidence=0.95,
            ocr_match_score=1.0,
            agreement_score=1.0,
            has_bbox=True,
            ocr_elements_count=5,
            evidence_found=False
        )
        self.assertLessEqual(result2["final_score"], 0.59)
        self.assertEqual(result2["confidence_label"], "Low")

        # 3. review_required=True and evidence_found=False should cap at 0.49
        result3 = self.engine.calculate_final_confidence(
            model_confidence=0.95,
            ocr_match_score=1.0,
            agreement_score=1.0,
            has_bbox=True,
            ocr_elements_count=5,
            review_required=True,
            evidence_found=False
        )
        self.assertLessEqual(result3["final_score"], 0.49)
        self.assertEqual(result3["confidence_label"], "Human Review Required")

    def test_nvidia_normalization_safety(self):
        import json
        client = NVIDIAClient()
        result_json = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "answer": "Yes",
                            "confidence": "n/a",
                            "reasoning": "This is raw reasoning key.",
                            "evidence_found": None,
                            "review_required": True
                        })
                    }
                }
            ]
        }
        normalized = client._normalize_response(result_json, 1)
        self.assertEqual(normalized.confidence, 0.0)
        self.assertEqual(normalized.reasoning_summary, "This is raw reasoning key.")
        self.assertEqual(normalized.evidence_found, True)
        self.assertEqual(normalized.review_required, True)

        result_json_percent = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "answer": "No",
                            "confidence": "85%",
                            "reasoning_summary": "Passed."
                        })
                    }
                }
            ]
        }
        normalized_percent = client._normalize_response(result_json_percent, 1)
        self.assertEqual(normalized_percent.confidence, 0.85)

    def test_page_ranker_boost_and_diagnostics(self):
        from core.page_ranker import PageRanker
        mock_repo = DocumentRepository(
            document_name="drawing.pdf",
            file_hash="12345",
            pages=[
                PageEntry(
                    page_number=1,
                    dimensions=PageDimension(width=1000, height=800),
                    image_path="p1.png",
                    preview_path="p1_prev.png",
                    page_hash="h1",
                    ocr_elements=[OCRElement(text="SD", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9, page_number=1)],
                    metadata={"page_type": "drawing"}
                ),
                PageEntry(
                    page_number=2,
                    dimensions=PageDimension(width=1000, height=800),
                    image_path="p2.png",
                    preview_path="p2_prev.png",
                    page_hash="h2",
                    ocr_elements=[OCRElement(text="SD", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9, page_number=2)],
                    metadata={"page_type": "legend"}
                )
            ],
            doc_intelligence={
                "legend_pages": [2],
                "page_types": [{"page": 1, "page_type": "drawing"}, {"page": 2, "page_type": "legend"}]
            }
        )
        store = EvidenceStore(mock_repo)
        rankings = PageRanker.rank_pages("How many supply diffusers?", "counting", store, mock_repo.doc_intelligence)
        self.assertEqual(rankings[0]["page"], 1)
        self.assertGreater(rankings[0]["score"], rankings[1]["score"])
        self.assertIn("ocr_keyword_matches", rankings[0])
        self.assertIn("equipment_matches", rankings[0])
        self.assertIn("symbol_matches", rankings[0])
        self.assertIn("final_score", rankings[0])

    def test_counting_legend_sheet_exclusions(self):
        from core.counting_engine import CountingEngine
        mock_repo = DocumentRepository(
            document_name="drawing.pdf",
            file_hash="12345",
            pages=[
                PageEntry(
                    page_number=1,
                    dimensions=PageDimension(width=1000, height=800),
                    image_path="p1.png",
                    preview_path="p1_prev.png",
                    page_hash="h1",
                    ocr_elements=[OCRElement(text="SD", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9, page_number=1)],
                    metadata={"page_type": "drawing"}
                ),
                PageEntry(
                    page_number=2,
                    dimensions=PageDimension(width=1000, height=800),
                    image_path="p2.png",
                    preview_path="p2_prev.png",
                    page_hash="h2",
                    ocr_elements=[
                        OCRElement(text="SD", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9, page_number=2),
                        OCRElement(text="Supply Diffuser", bbox=[0.1, 0.3, 0.2, 0.4], confidence=0.9, page_number=2)
                    ],
                    metadata={"page_type": "legend"}
                )
            ],
            doc_intelligence={
                "legend_pages": [2],
                "page_types": [{"page": 1, "page_type": "drawing"}, {"page": 2, "page_type": "legend"}]
            }
        )
        store = EvidenceStore(mock_repo)
        p_resp = InterpretationResponse(
            answer="3",
            confidence=0.9,
            page_number=1,
            bbox=[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]],
            evidence_found=True,
            review_required=False,
            reasoning_summary=""
        )
        audit = CountingEngine.audit_count("How many supply diffusers?", p_resp, None, store, [1, 2])
        self.assertEqual(audit["ocr_count"], 1)


if __name__ == "__main__":
    unittest.main()
