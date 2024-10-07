using Ukol2;
using static Ukol2.Program;

namespace test
{
    public class UnitTest1
    {
        public enum ParseIntOption
        {
            NONE,
            ALLOW_WHITESPACES,
            ALLOW_NEGATIVE_NUMBERS,
            IGNORE_INVALID_CHARACTERS
        }
        [Fact] // Ozna?uje jednotkový test
        public void ParseInt_ValidPositiveInput_ReturnsExpectedValue()
        {
            // Arrange
            string input = "123";

            // Act
            int result = Program.ParseInt(input);

            // Assert
            Assert.Equal(123, result);
        }

        [Fact]
        public void ParseInt_InvalidInput_ReturnsMinusOne()
        {
            // Arrange
            string input = "abc";

            // Act
            int result = Program.ParseInt(input);

            // Assert
            Assert.Equal(-1, result);
        }

        [Fact]
        public void TryParseInt_ValidNegativeInput_ReturnsTrue()
        {
            // Arrange
            string input = "-456";
            bool success = Program.TryParseInt(input, out int result);

            // Assert
            Assert.True(success);
            Assert.Equal(-456, result);
        }

        [Fact]
        public void TryParseInt_InvalidInput_ReturnsFalse()
        {
            // Arrange
            string input = "xyz";
            bool success = Program.TryParseInt(input, out int result);

            // Assert
            Assert.False(success);
        }

        [Fact]
        public void ParseIntOrNull_ValidPositiveInput_ReturnsExpectedValue()
        {
            // Arrange
            string input = "789";

            // Act
            int? result = Program.ParseIntOrNull(input);

            // Assert
            Assert.Equal(789, result);
        }

        [Fact]
        public void ParseIntOrNull_InvalidInput_ReturnsNull()
        {
            // Arrange
            string input = "not_a_number";

            // Act
            int? result = Program.ParseIntOrNull(input);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void TryParseInt_ValidInput_ReturnsTrue()
        {
            // Arrange
            string input = "123";
            ParseIntOption options = ParseIntOption.NONE;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.True(success);
            Assert.Equal(123, result);
        }

        [Fact]
        public void TryParseInt_Invalid_Input_ReturnsFalse()
        {
            // Arrange
            string input = "abc";
            ParseIntOption options = ParseIntOption.NONE;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.False(success);
        }

        [Fact]
        public void TryParseInt_AllowWhitespaces_ValidInput_ReturnsTrue()
        {
            // Arrange
            string input = " 1 2 3 ";
            ParseIntOption options = ParseIntOption.ALLOW_WHITESPACES;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.True(success);
            Assert.Equal(123, result);
        }

        [Fact]
        public void TryParseInt_AllowNegativeNumbers_ValidInput_ReturnsTrue()
        {
            // Arrange
            string input = "-456";
            ParseIntOption options = ParseIntOption.ALLOW_NEGATIVE_NUMBERS;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.True(success);
            Assert.Equal(-456, result);
        }

        [Fact]
        public void TryParseInt_AllowNegativeNumbers_NotAllowedNegative_ReturnsFalse()
        {
            // Arrange
            string input = "-456";
            ParseIntOption options = ParseIntOption.NONE;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.False(success);
        }

        [Fact]
        public void TryParseInt_IgnoreInvalidCharacters_ValidInputWithInvalidCharacters_ReturnsTrue()
        {
            // Arrange
            string input = "1a2b3";
            ParseIntOption options = ParseIntOption.IGNORE_INVALID_CHARACTERS;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.True(success);
            Assert.Equal(123, result);
        }

        [Fact]
        public void TryParseInt_CombinationOfOptions_ValidInputWithWhitespaceAndInvalidChars_ReturnsTrue()
        {
            // Arrange
            string input = " 1a 2 b3 ";
            ParseIntOption options = ParseIntOption.ALLOW_WHITESPACES | ParseIntOption.IGNORE_INVALID_CHARACTERS;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.True(success);
            Assert.Equal(123, result);
        }

        [Fact]
        public void TryParseInt_Combination_ValidNegativeInputWithInvalidChars_ReturnsTrue()
        {
            // Arrange
            string input = "-456";
            ParseIntOption options = ParseIntOption.ALLOW_NEGATIVE_NUMBERS;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.True(success);
            Assert.Equal(-456, result);
        }

        [Fact]
        public void TryParseInt_EmptyInput_ReturnsFalse()
        {
            // Arrange
            string input = "";
            ParseIntOption options = ParseIntOption.NONE;

            // Act
            bool success = Program.TryParseInt2(input, (Program.ParseIntOption)options, out int result);

            // Assert
            Assert.False(success);
        }
    }
}
