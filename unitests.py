import unittest
from unittest.mock import patch
from io import StringIO
from chatbot import chat

class ChatbotTestCase(unittest.TestCase):
    @patch('builtins.input', side_effect=['Hello', 'exit'])
    @patch('response.get_response', return_value='Hi, how can I help you?')
    def test_chat(self, mock_input, mock_get_response):
        expected_output = "Welcome to the Chatbot! Type 'exit' to end the conversation.\n" \
                          "User: Hello\n" \
                          "Chatbot: Hi, how can I help you?\n"

        with patch('sys.stdout', new=StringIO()) as fake_out:
            chat()

            self.assertEqual(fake_out.getvalue(), expected_output)



if __name__ == '__main__':
    unittest.main()
