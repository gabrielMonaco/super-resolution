import unittest
from unittest.mock import patch
from data import Data

class TestData(unittest.TestCase):
    def setUp(self):
        self.data = Data()

    @patch('data.os.path.join')
    @patch('data.Data._hr_images_dir')
    def test_hr_image_files(self, mock_hr_images_dir, mock_join):
        # Mock the return value of _hr_images_dir
        mock_hr_images_dir.return_value = '/path/to/images'

        # Set the image_ids attribute
        self.data.image_ids = [1, 2, 3]

        # Call the _hr_image_files method
        result = self.data._hr_image_files()

        # Assert that os.path.join was called with the correct arguments
        expected_calls = [mock_join.return_value for _ in self.data.image_ids]
        mock_join.assert_called_with('/path/to/images', mock_hr_images_dir.return_value)

        # Assert that the result is a list of file paths
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.data.image_ids))
        for file_path in result:
            self.assertIsInstance(file_path, str)

if __name__ == '__main__':
    unittest.main()