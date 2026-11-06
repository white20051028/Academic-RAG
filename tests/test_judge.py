"""判断层的纯函数测试。

**这个文件里没有任何网络调用。**
不连 SiliconFlow、不读 .env、不需要 API key——所以它能在 CI 里跑，
也能在没有网的电脑上跑。

`check_answerable()` 需要 client，这里用一个假的 client 代替真调用。
假 client 还能构造出「上游挂了你怎么办」这种真环境里不好复现的情况。
"""

import judge
import pytest

# ---------------------------------------------------------------- 闲聊识别


@pytest.mark.parametrize(
    "question",
    ["你好", "在吗", "谢谢", "你是谁", "你能干什么", "您好呀", "hello", "好的"],
)
def test_认出闲聊(question):
    assert judge.is_smalltalk(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "你好，转专业要什么条件",
        "我想问问绩点2.1能不能拿学位",
        "谢谢，那挂科了要重修吗",
    ],
)
def test_不误伤真问题(question):
    """问候词 + 实质内容，不能被当成闲聊。

    这是三分类里最危险的一类错误：把真问题当闲聊，用户会得到一段
    「我是教务问答」然后问题没了。
    """
    assert judge.is_smalltalk(question) is False


def test_空字符串当闲聊():
    assert judge.is_smalltalk("   ") is True


# ---------------------------------------------------------------- 判定解析


def test_能且给了依据():
    assert judge.parse_verdict("能 | 依据：学生可以在大二申请转专业") is True


def test_能不能没有依据():
    """只说了「能」但没摘出依据 —— 不算。

    模型经常这样偷懒：看到一点相关内容就说能。要求它摘依据，
    是逼它真的在资料里找过。
    """
    assert judge.parse_verdict("能") is False


def test_不能说():
    assert judge.parse_verdict("不能 | 原因：资料里没有这个信息") is False


def test_能但是其实是不能():
    """开头是「能」但整句是转折，且没给依据 —— 不能误判为能。"""
    assert judge.parse_verdict("能理解你的问题，但是资料里没有") is False


def test_空回答():
    assert judge.parse_verdict("") is False


# ---------------------------------------------------------------- prompt 构造


def test_probe_带上了问题和资料():
    probe = judge.build_probe("转专业要什么条件", "资料：可以转，需绩点3.0")
    assert "转专业要什么条件" in probe
    assert "可以转，需绩点3.0" in probe


def test_probe_资料超长会截断():
    """3000 字上限，防止把 5 条长资料全塞进去撑爆上下文。"""
    probe = judge.build_probe("问题", "长" * 5000)
    assert "长" * 3001 not in probe


# ---------------------------------------------------------------- 假 client

class _FakeClient:
    """假装是 openai.OpenAI，只实现 chat.completions.create。"""

    def __init__(self, reply=None, boom=False):
        self._reply = reply or ""
        self._boom = boom
        self.calls = 0

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                if outer._boom:
                    raise ConnectionError("假装上游连不上")

                class _Msg:
                    content = outer._reply

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_check_answerable_正常返回能():
    client = _FakeClient("能 | 依据：绩点 3.0 以上可以转")
    assert judge.check_answerable("能转专业吗", "资料", client, "fake-model") is True


def test_check_answerable_正常返回不能():
    client = _FakeClient("不能 | 原因：没这条规定")
    assert judge.check_answerable("能转专业吗", "资料", client, "fake-model") is False


def test_上游挂了时退化成能答():
    """判断层自己炸了，不能把整个问答也炸掉。

    策略是「宁可多答，不可误拒」：让用户看到答案和引用，
    他自己能核对；而直接拒答会让答案彻底消失。
    """
    client = _FakeClient(boom=True)
    assert judge.check_answerable("能转专业吗", "资料", client, "fake-model") is True
    assert client.calls == 1


def test_check_answerable_调的是传入的模型名():
    """模型名必须由调用方传，不能写死在模块里。

    写死就意味着线上用 A 模型、评测用 B 模型，数字对不上还查不出来。
    """
    seen = {}

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)

            class _Msg:
                content = "能 | 依据：x"

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    judge.check_answerable("q", "ctx", _Client(), "某个模型名")
    assert seen["model"] == "某个模型名"
    assert seen["temperature"] == 0
