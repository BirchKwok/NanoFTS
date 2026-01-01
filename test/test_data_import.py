"""
NanoFTS 数据导入测试

测试各种数据源导入功能：
- from_pandas() - 从 pandas DataFrame 导入
- from_polars() - 从 Polars DataFrame 导入
- from_arrow() - 从 PyArrow Table 导入
- from_parquet() - 从 Parquet 文件导入
- from_csv() - 从 CSV 文件导入
- from_json() - 从 JSON 文件导入
- from_dict() - 从字典列表导入
"""

import pytest
import os
from nanofts import create_engine


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_import.nfts")


@pytest.fixture
def engine(tmp_index_file):
    """创建测试引擎"""
    return create_engine(tmp_index_file, drop_if_exists=True)


@pytest.fixture
def sample_data():
    """示例数据"""
    return [
        {'id': 1, 'title': 'Hello World', 'content': 'This is a test document'},
        {'id': 2, 'title': '全文搜索', 'content': '支持多语言搜索'},
        {'id': 3, 'title': 'Python Document', 'content': 'Another test content'},
        {'id': 4, 'title': 'Mixed 混合', 'content': 'Both English and 中文'},
        {'id': 5, 'title': 'Search Engine', 'content': 'Fast and efficient'},
    ]


# ==================== from_pandas 测试 ====================

class TestFromPandas:
    """测试 from_pandas 方法"""
    
    def test_basic_import(self, engine, sample_data):
        """测试基本导入"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame(sample_data)
        count = engine.from_pandas(df, id_column='id')
        
        assert count == 5
        
        # 验证数据可搜索
        assert engine.search("hello").total_hits == 1
        assert engine.search("全文").total_hits == 1
        assert engine.search("test").total_hits == 2
    
    def test_custom_text_columns(self, engine):
        """测试指定文本列"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['A', 'B', 'C'],
            'content': ['Content 1', 'Content 2', 'Content 3'],
            'metadata': ['Meta 1', 'Meta 2', 'Meta 3']  # 不索引此列
        })
        
        count = engine.from_pandas(df, id_column='id', text_columns=['title', 'content'])
        
        assert count == 3
        
        # title 和 content 应该可搜索
        assert engine.search("content").total_hits == 3
        
        # metadata 不应该被索引
        assert engine.search("meta").total_hits == 0
    
    def test_different_id_column(self, engine):
        """测试不同的 ID 列名"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'doc_id': [100, 200, 300],
            'text': ['Document A', 'Document B', 'Document C']
        })
        
        count = engine.from_pandas(df, id_column='doc_id')
        
        assert count == 3
        
        result = engine.search("document")
        assert result.total_hits == 3
        assert 100 in result.to_list()
        assert 200 in result.to_list()
        assert 300 in result.to_list()
    
    def test_empty_dataframe(self, engine):
        """测试空 DataFrame"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame(columns=['id', 'title', 'content'])
        count = engine.from_pandas(df, id_column='id')
        
        assert count == 0
    
    def test_chinese_content(self, engine):
        """测试中文内容"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['北京市', '上海市', '广州市'],
            'content': ['首都城市', '经济中心', '南方门户']
        })
        
        count = engine.from_pandas(df, id_column='id')
        
        assert count == 3
        
        assert engine.search("北京").total_hits == 1
        assert engine.search("城市").total_hits == 1
    
    def test_numeric_content(self, engine):
        """测试数值内容（自动转换为字符串）"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['Product A', 'Product B', 'Product C'],
            'price': [100, 200, 300]
        })
        
        count = engine.from_pandas(df, id_column='id')
        
        assert count == 3
        
        # 数值被转换为字符串后可搜索
        assert engine.search("100").total_hits == 1
    
    def test_null_values(self, engine):
        """测试空值处理"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['Has Title', None, 'Another Title'],
            'content': ['Content 1', 'Content 2', None]
        })
        
        count = engine.from_pandas(df, id_column='id')
        
        assert count == 3
        
        # 应该能搜索到非空内容
        assert engine.search("title").total_hits == 2


# ==================== from_polars 测试 ====================

class TestFromPolars:
    """测试 from_polars 方法"""
    
    def test_basic_import(self, engine, sample_data):
        """测试基本导入"""
        pl = pytest.importorskip("polars")
        
        df = pl.DataFrame(sample_data)
        count = engine.from_polars(df, id_column='id')
        
        assert count == 5
        
        # 验证数据可搜索
        assert engine.search("hello").total_hits == 1
        assert engine.search("全文").total_hits == 1
    
    def test_custom_text_columns(self, engine):
        """测试指定文本列"""
        pl = pytest.importorskip("polars")
        
        df = pl.DataFrame({
            'id': [1, 2, 3],
            'title': ['Title 1', 'Title 2', 'Title 3'],
            'content': ['Content 1', 'Content 2', 'Content 3'],
            'private': ['Private 1', 'Private 2', 'Private 3']
        })
        
        count = engine.from_polars(df, id_column='id', text_columns=['title', 'content'])
        
        assert count == 3
        
        assert engine.search("title").total_hits == 3
        assert engine.search("private").total_hits == 0
    
    def test_different_id_column(self, engine):
        """测试不同的 ID 列名"""
        pl = pytest.importorskip("polars")
        
        df = pl.DataFrame({
            'doc_id': [10, 20, 30],
            'text': ['Polars Doc A', 'Polars Doc B', 'Polars Doc C']
        })
        
        count = engine.from_polars(df, id_column='doc_id')
        
        assert count == 3
        
        result = engine.search("polars")
        assert result.total_hits == 3
        assert 10 in result.to_list()
    
    def test_large_dataframe(self, engine):
        """测试大数据量"""
        pl = pytest.importorskip("polars")
        
        df = pl.DataFrame({
            'id': list(range(1000)),
            'content': [f'Document content number {i}' for i in range(1000)]
        })
        
        count = engine.from_polars(df, id_column='id')
        
        assert count == 1000
        
        result = engine.search("document")
        assert result.total_hits == 1000


# ==================== from_arrow 测试 ====================

class TestFromArrow:
    """测试 from_arrow 方法"""
    
    def test_basic_import(self, engine, sample_data):
        """测试基本导入"""
        pa = pytest.importorskip("pyarrow")
        
        table = pa.Table.from_pydict({
            'id': [d['id'] for d in sample_data],
            'title': [d['title'] for d in sample_data],
            'content': [d['content'] for d in sample_data]
        })
        
        count = engine.from_arrow(table, id_column='id')
        
        assert count == 5
        
        # 验证数据可搜索
        assert engine.search("hello").total_hits == 1
    
    def test_custom_text_columns(self, engine):
        """测试指定文本列"""
        pa = pytest.importorskip("pyarrow")
        
        table = pa.Table.from_pydict({
            'id': [1, 2, 3],
            'indexable': ['Index A', 'Index B', 'Index C'],
            'skip': ['Skip A', 'Skip B', 'Skip C']
        })
        
        count = engine.from_arrow(table, id_column='id', text_columns=['indexable'])
        
        assert count == 3
        
        assert engine.search("index").total_hits == 3
        assert engine.search("skip").total_hits == 0
    
    def test_from_pandas_conversion(self, engine):
        """测试从 pandas 转换的 Arrow Table"""
        pd = pytest.importorskip("pandas")
        pa = pytest.importorskip("pyarrow")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['Arrow A', 'Arrow B', 'Arrow C']
        })
        
        table = pa.Table.from_pandas(df)
        count = engine.from_arrow(table, id_column='id')
        
        assert count == 3
        
        result = engine.search("arrow")
        assert result.total_hits == 3


# ==================== from_parquet 测试 ====================

class TestFromParquet:
    """测试 from_parquet 方法"""
    
    def test_basic_import(self, engine, sample_data, tmp_path):
        """测试基本导入"""
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        
        # 创建 Parquet 文件
        table = pa.Table.from_pydict({
            'id': [d['id'] for d in sample_data],
            'title': [d['title'] for d in sample_data],
            'content': [d['content'] for d in sample_data]
        })
        
        parquet_path = tmp_path / "test.parquet"
        pq.write_table(table, parquet_path)
        
        # 导入
        count = engine.from_parquet(parquet_path, id_column='id')
        
        assert count == 5
        
        # 验证数据可搜索
        assert engine.search("hello").total_hits == 1
        assert engine.search("全文").total_hits == 1
    
    def test_custom_text_columns(self, engine, tmp_path):
        """测试指定文本列"""
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        
        table = pa.Table.from_pydict({
            'id': [1, 2, 3],
            'searchable': ['Search A', 'Search B', 'Search C'],
            'hidden': ['Hidden A', 'Hidden B', 'Hidden C']
        })
        
        parquet_path = tmp_path / "test.parquet"
        pq.write_table(table, parquet_path)
        
        count = engine.from_parquet(parquet_path, id_column='id', text_columns=['searchable'])
        
        assert count == 3
        
        assert engine.search("search").total_hits == 3
        assert engine.search("hidden").total_hits == 0
    
    def test_string_path(self, engine, sample_data, tmp_path):
        """测试字符串路径"""
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        
        table = pa.Table.from_pydict({
            'id': [d['id'] for d in sample_data],
            'title': [d['title'] for d in sample_data]
        })
        
        parquet_path = str(tmp_path / "test_string_path.parquet")
        pq.write_table(table, parquet_path)
        
        count = engine.from_parquet(parquet_path, id_column='id')
        
        assert count == 5


# ==================== from_csv 测试 ====================

class TestFromCSV:
    """测试 from_csv 方法"""
    
    def test_basic_import(self, engine, sample_data, tmp_path):
        """测试基本导入"""
        pd = pytest.importorskip("pandas")
        
        # 创建 CSV 文件
        df = pd.DataFrame(sample_data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # 导入
        count = engine.from_csv(csv_path, id_column='id')
        
        assert count == 5
        
        # 验证数据可搜索
        assert engine.search("hello").total_hits == 1
        assert engine.search("全文").total_hits == 1
    
    def test_custom_text_columns(self, engine, tmp_path):
        """测试指定文本列"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['CSV Title 1', 'CSV Title 2', 'CSV Title 3'],
            'ignore': ['Ignore 1', 'Ignore 2', 'Ignore 3']
        })
        
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        count = engine.from_csv(csv_path, id_column='id', text_columns=['title'])
        
        assert count == 3
        
        assert engine.search("csv").total_hits == 3
        assert engine.search("ignore").total_hits == 0
    
    def test_csv_options(self, engine, tmp_path):
        """测试 CSV 选项"""
        pd = pytest.importorskip("pandas")
        
        # 创建使用分号分隔的 CSV
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['Semicolon 1', 'Semicolon 2', 'Semicolon 3']
        })
        
        csv_path = tmp_path / "test_semicolon.csv"
        df.to_csv(csv_path, index=False, sep=';')
        
        # 使用 sep 选项导入
        count = engine.from_csv(csv_path, id_column='id', sep=';')
        
        assert count == 3
        
        result = engine.search("semicolon")
        assert result.total_hits == 3
    
    def test_encoding(self, engine, tmp_path):
        """测试编码选项"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['中文标题1', '中文标题2', '中文标题3']
        })
        
        csv_path = tmp_path / "test_utf8.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        count = engine.from_csv(csv_path, id_column='id', encoding='utf-8')
        
        assert count == 3
        
        result = engine.search("中文")
        assert result.total_hits == 3


# ==================== from_json 测试 ====================

class TestFromJSON:
    """测试 from_json 方法"""
    
    def test_basic_import(self, engine, sample_data, tmp_path):
        """测试基本导入"""
        pd = pytest.importorskip("pandas")
        
        # 创建 JSON 文件
        df = pd.DataFrame(sample_data)
        json_path = tmp_path / "test.json"
        df.to_json(json_path, orient='records')
        
        # 导入
        count = engine.from_json(json_path, id_column='id')
        
        assert count == 5
        
        # 验证数据可搜索
        assert engine.search("hello").total_hits == 1
    
    def test_json_lines(self, engine, tmp_path):
        """测试 JSON Lines 格式"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['JSON Line 1', 'JSON Line 2', 'JSON Line 3']
        })
        
        jsonl_path = tmp_path / "test.jsonl"
        df.to_json(jsonl_path, orient='records', lines=True)
        
        count = engine.from_json(jsonl_path, id_column='id', lines=True)
        
        assert count == 3
        
        result = engine.search("json")
        assert result.total_hits == 3
    
    def test_custom_text_columns(self, engine, tmp_path):
        """测试指定文本列"""
        pd = pytest.importorskip("pandas")
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'include': ['Include A', 'Include B', 'Include C'],
            'exclude': ['Exclude A', 'Exclude B', 'Exclude C']
        })
        
        json_path = tmp_path / "test.json"
        df.to_json(json_path, orient='records')
        
        count = engine.from_json(json_path, id_column='id', text_columns=['include'])
        
        assert count == 3
        
        assert engine.search("include").total_hits == 3
        assert engine.search("exclude").total_hits == 0


# ==================== from_dict 测试 ====================

class TestFromDict:
    """测试 from_dict 方法"""
    
    def test_basic_import(self, engine, sample_data):
        """测试基本导入"""
        count = engine.from_dict(sample_data, id_column='id')
        
        assert count == 5
        
        # 验证数据可搜索
        assert engine.search("hello").total_hits == 1
        assert engine.search("全文").total_hits == 1
    
    def test_custom_text_columns(self, engine):
        """测试指定文本列"""
        data = [
            {'id': 1, 'title': 'Dict Title 1', 'secret': 'Secret 1'},
            {'id': 2, 'title': 'Dict Title 2', 'secret': 'Secret 2'},
            {'id': 3, 'title': 'Dict Title 3', 'secret': 'Secret 3'},
        ]
        
        count = engine.from_dict(data, id_column='id', text_columns=['title'])
        
        assert count == 3
        
        assert engine.search("dict").total_hits == 3
        assert engine.search("secret").total_hits == 0
    
    def test_empty_list(self, engine):
        """测试空列表"""
        count = engine.from_dict([], id_column='id')
        
        assert count == 0
    
    def test_different_id_column(self, engine):
        """测试不同的 ID 列名"""
        data = [
            {'doc_id': 100, 'text': 'Dict Doc A'},
            {'doc_id': 200, 'text': 'Dict Doc B'},
            {'doc_id': 300, 'text': 'Dict Doc C'},
        ]
        
        count = engine.from_dict(data, id_column='doc_id')
        
        assert count == 3
        
        result = engine.search("dict")
        assert result.total_hits == 3
        assert 100 in result.to_list()
    
    def test_missing_fields(self, engine):
        """测试缺失字段"""
        data = [
            {'id': 1, 'title': 'Has All Fields', 'content': 'Content 1'},
            {'id': 2, 'title': 'Missing Content'},  # 缺少 content
            {'id': 3, 'content': 'Missing Title'},  # 缺少 title
        ]
        
        count = engine.from_dict(data, id_column='id')
        
        assert count == 3
        
        # 应该能搜索到有值的字段
        assert engine.search("missing").total_hits == 2


# ==================== 综合测试 ====================

class TestMixedImport:
    """综合导入测试"""
    
    def test_multiple_imports(self, engine):
        """测试多次导入"""
        # 第一次导入
        data1 = [
            {'id': 1, 'content': 'First batch document 1'},
            {'id': 2, 'content': 'First batch document 2'},
        ]
        count1 = engine.from_dict(data1, id_column='id')
        assert count1 == 2
        
        # 第二次导入
        data2 = [
            {'id': 3, 'content': 'Second batch document 3'},
            {'id': 4, 'content': 'Second batch document 4'},
        ]
        count2 = engine.from_dict(data2, id_column='id')
        assert count2 == 2
        
        # 验证两批数据都可搜索
        result = engine.search("batch")
        assert result.total_hits == 4
        
        result = engine.search("first")
        assert result.total_hits == 2
        
        result = engine.search("second")
        assert result.total_hits == 2
    
    def test_import_and_search_workflow(self, engine):
        """测试导入和搜索工作流"""
        # 导入数据
        data = [
            {'id': 1, 'title': 'Machine Learning', 'content': 'Deep learning algorithms'},
            {'id': 2, 'title': 'Natural Language', 'content': 'Text processing NLP'},
            {'id': 3, 'title': 'Computer Vision', 'content': 'Image recognition CNN'},
            {'id': 4, 'title': 'Reinforcement Learning', 'content': 'Agent policy optimization'},
        ]
        
        engine.from_dict(data, id_column='id')
        
        # 测试各种搜索
        assert engine.search("learning").total_hits == 2
        assert engine.search("machine").total_hits == 1
        
        # 测试 AND 搜索
        result = engine.search_and(["learning", "machine"])
        assert result.total_hits == 1
        assert 1 in result.to_list()
        
        # 测试 OR 搜索
        result = engine.search_or(["machine", "computer"])
        assert result.total_hits == 2
    
    def test_import_persistence(self, tmp_path):
        """测试导入后持久化"""
        index_file = str(tmp_path / "persist_test.nfts")
        
        # 创建引擎并导入
        engine1 = create_engine(index_file, drop_if_exists=True)
        data = [
            {'id': 1, 'content': 'Persistent data 1'},
            {'id': 2, 'content': 'Persistent data 2'},
        ]
        engine1.from_dict(data, id_column='id')
        del engine1
        
        # 重新打开并验证
        engine2 = create_engine(index_file)
        result = engine2.search("persistent")
        
        assert result.total_hits == 2
        assert 1 in result.to_list()
        assert 2 in result.to_list()


# ==================== 边缘情况测试 ====================

class TestEdgeCases:
    """边缘情况测试"""
    
    def test_very_long_content(self, engine):
        """测试超长内容"""
        data = [
            {'id': 1, 'content': 'word ' * 10000},  # 50000 字符
        ]
        
        count = engine.from_dict(data, id_column='id')
        
        assert count == 1
        
        result = engine.search("word")
        assert 1 in result.to_list()
    
    def test_special_characters(self, engine):
        """测试特殊字符"""
        data = [
            {'id': 1, 'content': 'Special @#$% characters!'},
            {'id': 2, 'content': 'Email test@example.com'},
            {'id': 3, 'content': 'URL https://example.com'},
        ]
        
        count = engine.from_dict(data, id_column='id')
        
        assert count == 3
        
        assert engine.search("special").total_hits == 1
        assert engine.search("email").total_hits == 1
    
    def test_unicode_content(self, engine):
        """测试 Unicode 内容"""
        data = [
            {'id': 1, 'content': '日本語テスト'},
            {'id': 2, 'content': '한국어 테스트'},
            {'id': 3, 'content': 'Emoji 🎉🎊'},
        ]
        
        count = engine.from_dict(data, id_column='id')
        
        assert count == 3
    
    def test_large_id_values(self, engine):
        """测试大 ID 值"""
        data = [
            {'id': 1000000, 'content': 'Large ID 1'},
            {'id': 2000000, 'content': 'Large ID 2'},
            {'id': 2147483647, 'content': 'Max int ID'},  # 最大 32 位整数
        ]
        
        count = engine.from_dict(data, id_column='id')
        
        assert count == 3
        
        result = engine.search("large")
        assert 1000000 in result.to_list()
        assert 2000000 in result.to_list()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


