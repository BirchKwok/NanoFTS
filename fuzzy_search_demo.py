#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NanoFTS 模糊搜索功能演示

展示如何使用新增的模糊搜索功能，包括配置优化和最佳实践
"""

from nanofts import FullTextSearch

def main():
    print("🚀 NanoFTS 模糊搜索功能演示")
    print("=" * 50)
    
    # 初始化搜索引擎，使用优化的模糊搜索配置
    fts = FullTextSearch(
        index_dir=None,  # 内存模式，更快的演示
        fuzzy_threshold=0.5,  # 相似度阈值 (0.0-1.0)
        fuzzy_max_distance=2   # 最大编辑距离
    )
    
    # 添加测试文档
    test_docs = [
        {"title": "苹果产品", "content": "苹果iPhone手机是优秀的智能设备"},
        {"title": "华为技术", "content": "华为手机和5G技术领先全球"},
        {"title": "编程教程", "content": "Python编程技术教程和实例"},
        {"title": "开发指南", "content": "软件开发最佳实践指南"},
    ]
    
    print("📚 添加测试文档...")
    for i, doc in enumerate(test_docs, 1):
        fts.add_document(i, doc)
    fts.flush()
    
    print(f"✅ 成功添加 {len(test_docs)} 个文档\n")
    
    # 演示各种搜索场景
    print("🔍 搜索演示")
    print("-" * 30)
    
    test_cases = [
        ("苹果", "精确匹配"),
        ("苹檎", "错别字 - 模糊搜索"),
        ("编成", "错别字 - 模糊搜索"),
        ("华维", "拼音相似 - 模糊搜索"),
    ]
    
    for query, description in test_cases:
        print(f"🔎 搜索: '{query}' ({description})")
        
        # 精确搜索
        exact_results = fts.search(query, enable_fuzzy=False)
        print(f"   精确搜索: {len(exact_results)} 个结果")
        
        # 模糊搜索
        fuzzy_results = fts.search(query, enable_fuzzy=True, min_results=1)
        print(f"   模糊搜索: {len(fuzzy_results)} 个结果")
        
        # 显示找到的文档
        if len(fuzzy_results) > 0:
            for doc_id in list(fuzzy_results)[:2]:
                doc = test_docs[doc_id - 1]
                print(f"     📄 [{doc_id}] {doc['title']}")
        print()
    
    # 演示便捷方法
    print("🎯 便捷的模糊搜索方法")
    print("-" * 30)
    results = fts.fuzzy_search("技朮")  # 直接调用模糊搜索
    print(f"使用 fuzzy_search('技朮'): {len(results)} 个结果")
    
    # 演示配置管理
    print("\n⚙️ 配置管理")
    print("-" * 30)
    config = fts.get_fuzzy_config()
    print(f"当前配置: {config}")
    
    # 调整配置以获得更严格的匹配
    fts.set_fuzzy_config(fuzzy_threshold=0.8, fuzzy_max_distance=1)
    print("设置更严格的配置...")
    new_config = fts.get_fuzzy_config()
    print(f"新配置: {new_config}")
    
    print("\n✨ 模糊搜索功能特点:")
    print("• 🧠 智能启用：只在精确搜索结果不足时自动启用")
    print("• ⚡ 零I/O开销：完全在内存中进行，不影响磁盘性能") 
    print("• 🏃 高效缓存：重复查询使用缓存，避免重复计算")
    print("• 🔧 灵活配置：可根据需求调整相似度阈值和编辑距离")
    print("• 🌏 中英支持：同时支持中文和英文的模糊匹配")
    print("• 📊 适用场景：搜索纠错、智能推荐、输入法辅助等")

if __name__ == "__main__":
    main()