# Chat-Bot 代码重构计划 (TDD 驱动)

## 一、问题汇总

### 1. 代码风格问题

| 优先级 | 文件 | 问题 | 影响 |
|--------|------|------|------|
| **P0** | `audio_converter.py` | `convert_audio_format()` 119行，严重超标 | 难以维护和测试 |
| **P0** | `base_protocol.py` | `_handle_config_set()` 67行 | 职责过多 |
| **P0** | `audio_handler.py` | `_check_and_process()` 52行 | 逻辑复杂 |
| **P1** | `chat_engine.py` | `initialize()` 58行 | 职责混合 |
| **P1** | 多文件 | 魔法数字和硬编码路径 | 配置不灵活 |

### 2. 类型注解问题

| 优先级 | 问题 | 数量 |
|--------|------|------|
| **P0** | 缺少 `-> None` 返回值注解 | 50+ 方法 |
| **P1** | 实例属性缺少类型注解 | 30+ 属性 |
| **P2** | 使用旧式类型语法 (`Optional` vs `|`) | 全项目 |

### 3. 错误处理问题

| 优先级 | 文件 | 问题 |
|--------|------|------|
| **P1** | `audio_converter.py` | 异常被静默吞掉 (debug 日志) |
| **P1** | `silero_vad_adapter.py` | VAD 失败返回 False 可能掩盖错误 |
| **P2** | 多文件 | 过于宽泛的 `except Exception` |

### 4. 测试覆盖缺口

| 优先级 | 模块 | 状态 |
|--------|------|------|
| **P0** | `ConversationOrchestrator` | 完全无测试 (核心对话流程) |
| **P0** | `ConversationManager` | 完全无测试 (会话管理) |
| **P0** | `ConfigManager` | 完全无测试 (~350行) |
| **P1** | `BaseProtocol` config handlers | 部分缺失 |
| **P1** | `ChatEngine` shutdown/health | 部分缺失 |

### 5. 架构问题

| 优先级 | 问题 | 位置 |
|--------|------|------|
| **P0** | 全局状态 `AppContext._modules` | `app_context.py` |
| **P1** | `BaseProtocol` 职责过多 (443行) | `base_protocol.py` |
| **P1** | 配置管理两套机制 | `config_loader.py` + `config_manager.py` |
| **P2** | 日志格式不一致 | 多文件 |

---

## 二、TDD 重构任务清单

### Phase 1: 测试补全 (Week 1) ✅ 完成

#### Task 1.1: ConversationOrchestrator 测试 ✅
```
文件: backend/tests/unit/core/conversation/test_orchestrator.py
```

**测试用例:** (37 个测试已实现)
- [x] `test_trigger_conversation_calls_llm_and_tts`
- [x] `test_process_with_tts_splits_sentences`
- [x] `test_interrupt_stops_generation`
- [x] `test_on_input_result_concatenates_text`
- [x] `test_background_task_cleanup_on_stop`

#### Task 1.2: ConversationManager 测试 ✅
```
文件: backend/tests/unit/core/session/test_conversation_manager.py
```

**测试用例:** (7 个测试已实现)
- [x] `test_create_handler_success`
- [x] `test_create_handler_idempotent`
- [x] `test_destroy_handler_cleans_resources`
- [x] `test_destroy_all_handlers_concurrent_safe`
- [x] `test_get_handler_returns_none_if_not_exists`

#### Task 1.3: ConfigManager 测试 ✅
```
文件: backend/tests/unit/utils/test_config_manager.py
```

**测试用例:** (32 个测试已实现)
- [x] `test_get_config_full`
- [x] `test_get_config_by_section`
- [x] `test_update_config_deep_merge`
- [x] `test_mask_sensitive_fields_recursive`
- [x] `test_unmask_sensitive_fields_restores_values`
- [x] `test_validate_config_rejects_invalid`
- [x] `test_cache_invalidation`

---

### Phase 2: 代码重构 (Week 2) ✅ 完成

#### Task 2.1: 拆分 `convert_audio_format()` ✅
```
文件: backend/utils/audio_converter.py
```

**已完成重构:**
- `convert_audio_format()` - 主入口策略选择器
- `_load_audio_segment()` - 加载音频数据
- `_apply_audio_transformations()` - 应用采样率、通道、位深度转换
- `_segment_to_numpy()` - 转换为 NumPy 数组
- `_convert_to_output_format()` - 转换为目标格式
- `_convert_with_pydub()` - pydub 转换主逻辑

#### Task 2.2: 拆分 `BaseProtocol` ✅
```
文件: backend/core/handlers/config_handler.py (已创建)
     backend/core/handlers/status_handler.py (已创建)
     backend/tests/unit/core/handlers/test_config_handler.py (9 个测试)
     backend/tests/unit/core/handlers/test_status_handler.py (6 个测试)
```

**已完成:**
- `ConfigHandler` 类 - 处理配置获取和更新
- `StatusHandler` 类 - 处理模块状态查询
- 完整的单元测试覆盖

#### Task 2.3: 提取常量和配置 ✅
```
文件: backend/core/constants.py (已创建)
```

**已包含常量:**
- Audio Constants (AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, etc.)
- Buffer Constants
- Noise Reduction Constants
- VAD Constants
- WebSocket Constants
- Session Constants
- Default Paths

---

### Phase 3: 类型注解完善 (Week 3) ✅ 完成

#### Task 3.1: 添加返回值类型注解 ✅

**已完成文件:**
1. `backend/core/interfaces/base_protocol.py` - 所有方法添加 `-> None` 等返回类型
2. `backend/core/handlers/config_handler.py` - 完整类型注解
3. `backend/core/handlers/status_handler.py` - 完整类型注解

#### Task 3.2: 添加实例属性类型注解 ✅

**已完成文件:**
1. `backend/adapters/llm/langchain_llm_adapter.py` - 所有实例属性添加类型注解，使用 Python 3.10+ 语法
2. `backend/adapters/asr/funasr_sensevoice_adapter.py` - 所有实例属性添加类型注解
3. `backend/adapters/vad/silero_vad_adapter.py` - 所有实例属性添加类型注解
4. `backend/adapters/tts/edge_tts_adapter.py` - 所有实例属性添加类型注解

---

### Phase 4: 架构优化 (Week 4) ✅ 完成

#### Task 4.1: 移除全局状态 ✅

**已完成:**
1. 创建依赖注入容器 `backend/core/di/container.py`
   - 支持按名称/类型注册和解析依赖
   - 支持工厂函数注册
   - 线程安全，支持 clone() 用于测试隔离
2. 重构 `ChatEngine` 使用 Container
   - 在 __init__ 中创建 Container 实例
   - 在 initialize() 中注册所有模块到 Container
   - 添加 get_container() 方法
   - 保持 AppContext 向后兼容
3. 重构 `BaseProtocol` 支持依赖注入
   - 添加可选 module_provider 参数
   - 优先使用注入的 provider，回退到 AppContext
4. 完整的单元测试覆盖 (10 个测试)

**新增文件:**
- `backend/core/di/__init__.py`
- `backend/core/di/container.py`
- `backend/tests/unit/core/di/__init__.py`
- `backend/tests/unit/core/di/test_container.py`

#### Task 4.2: 统一日志格式 ✅

**已完成:**
- 在 `BaseModule` 中添加了 `module_type` 属性
- 更新了 `_log_prefix` 属性使用统一格式 `[ModuleType/ModuleID]`
- 日志辅助方法 (`log_info`, `log_debug`, `log_warning`, `log_error`) 自动添加前缀

---

## 三、验收标准

### 代码质量指标

| 指标 | 目标 |
|------|------|
| 单个方法行数 | ≤ 50 行 |
| 类型注解覆盖率 | ≥ 95% |
| 单元测试覆盖率 | ≥ 80% |
| 集成测试场景 | ≥ 10 个关键流程 |

### TDD 检查点

每个任务完成前必须:
1. ✅ 先写测试用例
2. ✅ 测试失败 (红)
3. ✅ 实现代码使测试通过 (绿)
4. ✅ 重构代码 (保持测试通过)
5. ✅ 代码审查

---

## 四、执行顺序

```
Week 1: Phase 1 (测试补全)
    └── Task 1.1 → Task 1.2 → Task 1.3

Week 2: Phase 2 (代码重构)
    └── Task 2.1 → Task 2.2 → Task 2.3

Week 3: Phase 3 (类型注解)
    └── Task 3.1 → Task 3.2

Week 4: Phase 4 (架构优化)
    └── Task 4.1 → Task 4.2
```

---

## 五、风险与注意事项

1. **向后兼容**: 重构 `BaseProtocol` 时需保持 API 兼容
2. **测试隔离**: 移除全局状态后需更新所有测试的 fixture
3. **渐进式重构**: 每个任务应可独立合并，避免大型 PR
4. **文档同步**: 更新 API 文档和使用示例

---

*生成时间: 2024-02-11*
*基于 5 个并行分析 Agent 的综合报告*
