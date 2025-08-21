import sys

class DNACompressor:
    def __init__(self, dna_sequence: str):
        self.dna_sequence = dna_sequence.upper()
        self._compressed_sequence = self._compress()

    def _compress(self):
        """Приватный метод для сжатия последовательности ДНК в бинарный формат."""
        nucleotide_map = {
            'A': 0b00,
            'C': 0b01,
            'G': 0b10,
            'T': 0b11
        }
        compressed = 0
        for nucleotide in self.dna_sequence:
            compressed <<= 2  # Сдвиг на 2 бита влево
            compressed |= nucleotide_map[nucleotide]  # Добавление закодированного нуклеотида
        return compressed

    def decompress(self):
        """Распаковка сжатой последовательности ДНК в строковый формат."""
        reverse_nucleotide_map = {
            0b00: 'A',
            0b01: 'C',
            0b10: 'G',
            0b11: 'T'
        }
        decompressed = []
        compressed = self._compressed_sequence
        for _ in range(len(self.dna_sequence)):
            decompressed.append(reverse_nucleotide_map[compressed & 0b11])  # Последние 2 бита
            compressed >>= 2  # Сдвиг вправо
        return ''.join(decompressed[::-1])  # Обратный порядок

    def __str__(self):
        """Возвращает распакованную строку."""
        return self.decompress()

# Функция тестирования
def test_compression(sequence_lengths):
    import random

    for length in sequence_lengths:
        # Генерация случайной последовательности
        dna_sequence = ''.join(random.choices('ACGT', k=length))

        # Создание экземпляра класса
        compressor = DNACompressor(dna_sequence)

        # Расчет размера оригинальной и сжатой последовательности
        original_size = sys.getsizeof(dna_sequence)
        compressed_size = sys.getsizeof(compressor._compressed_sequence)

        # Проверка распаковки
        assert compressor.decompress() == dna_sequence, "Ошибка при распаковке!"

        print(f"Длина: {length} символов")
        print(f"Оригинальный размер: {original_size} байт")
        print(f"Сжатый размер: {compressed_size} байт")
        print("-" * 30)

# Тестирование на последовательностях различной длины
sequence_lengths = [1000, 100_000, 1_000_000, 10_000_000]
test_compression(sequence_lengths)
