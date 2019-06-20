# -*- coding:utf-8 -*-

# base
# https://linux.thai.net/~thep/datrie/datrie.html
# http://jorbe.sinaapp.com/2014/05/11/datrie/
# http://www.hankcs.com/program/java/%E5%8F%8C%E6%95%B0%E7%BB%84trie%E6%A0%91doublearraytriejava%E5%AE%9E%E7%8E%B0.html
# (komiya-atsushi/darts-java | 先建立Trie树，再构造DAT，为siblings先找到合适的空间)
# https://blog.csdn.net/kissmile/article/details/47417277
# http://nark.cc/p/?p=1480
# https://github.com/midnight2104/midnight2104.github.io/blob/58b5664b3e16968dd24ac5b1b3f99dc21133b8c4/_posts/2018-8-8-%E5%8F%8C%E6%95%B0%E7%BB%84Trie%E6%A0%91(DoubleArrayTrie).md

# 不需要构造真正的Trie树，直接用字符串，构造对应node，因为words是排过序的
# todo : error info
# todo : performance test
# todo : resize
# warning: code=0表示叶子节点可能会有隐患(正常词汇的情况下是ok的)
# 修正: 由于想要回溯字符串的效果，叶子节点和base不能重合(这样叶子节点可以继续记录其他值比如频率)，叶子节点code: 0->-1
# 但是如此的话，叶子节点可能会与正常节点冲突？ 找begin的使用应该是考虑到的？
# from __future__ import print_function
class DATrie(object):
    class Node(object):

        def __init__(self, code, depth, left, right):
            self.code = code
            self.depth = depth
            self.left = left
            self.right = right

    def __init__(self):
        self.MAX_SIZE = 2097152  # 65536 * 32
        self.base = [0] * self.MAX_SIZE
        self.check = [-1] * self.MAX_SIZE  # -1 表示空
        self.used = [False] * self.MAX_SIZE
        self.nextCheckPos = 0  # 详细 见后面->当数组某段使用率达到某个值时记录下可用点，以便下次不再使用
        self.size = 0  # 记录总共用到的空间

    # 需要改变size的时候调用，这里只能用于build之前。cuz没有打算复制数据.
    def resize(self, size):
        self.MAX_SIZE = size
        self.base = [0] * self.MAX_SIZE
        self.check = [-1] * self.MAX_SIZE
        self.used = [False] * self.MAX_SIZE

    # 先决条件是self.words ordered 且没有重复
    # siblings至少会有一个
    def fetch(self, parent):  ###获取parent的孩子，存放在siblings中，并记录下其左右截至
        depth = parent.depth

        siblings = []  # size == parent.right-parent.left
        i = parent.left
        while i < parent.right:  # 遍历所有子节点，right-left+1个单词
            s = self.words[i][depth:]  # 词的后半部分
            if s == '':
                siblings.append(
                    self.Node(code=-1, depth=depth + 1, left=i, right=i + 1))  # 叶子节点
            else:
                c = ord(s[0])  # 字符串中每个汉字占用3个字符（code,实际也就当成符码），将每个字符转为数字 ，树实际是用这些数字构建的
                # print type(s[0]),c
                if siblings == [] or siblings[-1].code != c:
                    siblings.append(
                        self.Node(code=c, depth=depth + 1, left=i, right=i + 1))  # 新建节点
                else:  # siblings[-1].code == c
                    siblings[-1].right += 1  # 已经是排过序的可以直接计数+1
            i += 1
        # siblings
        return siblings

    # 在insert之前，认为可以先排序词汇，对base的分配检查应该是有利的
    # 先构建树，再构建DAT，再销毁树
    def build(self, words):
        words = sorted(list(set(words)))  # 去重排序
        # for word in words:print word.decode('utf-8')
        self.words = words
        # todo: 销毁_root
        _root = self.Node(code=0, depth=0, left=0, right=len(self.words))  # 增加第一个节点
        self.base[0] = 1
        siblings = self.fetch(_root)
        # for ii in  words: print ii.decode('utf-8')
        # print 'siblings len',len(siblings)
        # for i in siblings: print i.code
        self.insert(siblings, 0)  # 插入根节点的第一层孩子
        # while False:  # 利用队列来实现非递归构造
        # pass
        del self.words
        print("DATrie builded.")

    def insert(self, siblings, parent_base_idx):
        """ parent_base_idx为父节点base index, siblings为其子节点们 """
        # 暂时按komiya-atsushi/darts-java的方案
        # 总的来讲是从0开始分配beigin]
        # self.used[parent_base_idx] = True

        begin = 0
        pos = max(siblings[0].code + 1, self.nextCheckPos) - 1  # 从第一个孩子的字符码位置开始找，因为排过序，前面的都已经使用
        nonzero_num = 0  # 非零统计
        first = 0

        begin_ok_flag = False  # 找合适的begin
        while not begin_ok_flag:
            pos += 1
            if pos >= self.MAX_SIZE:
                raise Exception("no room, may be resize it.")
            if self.check[pos] != -1 or self.used[pos]:  # check——check数组，used——占用标记，表明pos位置已经占用
                nonzero_num += 1  # 已被使用
                continue
            elif first == 0:
                self.nextCheckPos = pos  # 第一个可以使用的位置，记录？仅执行一遍
                first = 1

            begin = pos - siblings[0].code  # 第一个孩子节点对应的begin

            if begin + siblings[-1].code >= self.MAX_SIZE:
                raise Exception("no room, may be resize it.")

            if self.used[begin]:  # 该位置已经占用
                continue

            if len(siblings) == 1:  # 只有一个节点
                begin_ok_flag = True
                break

            for sibling in siblings[1:]:
                if self.check[begin + sibling.code] == -1 and self.used[
                    begin + sibling.code] is False:  # 对于sibling，begin位置可用
                    begin_ok_flag = True
                else:
                    begin_ok_flag = False  # 用一个不可用，则begin不可用
                    break

        # 得到合适的begin

        # -- Simple heuristics --
        # if the percentage of non-empty contents in check between the
        # index 'next_check_pos' and 'check' is greater than some constant value
        # (e.g. 0.9), new 'next_check_pos' index is written by 'check'.

        # 从位置 next_check_pos 开始到 pos 间，如果已占用的空间在95%以上，下次插入节点时，直接从 pos 位置处开始查找成功获得这一层节点的begin之后得到，影响下一次执行insert时的查找效率
        if (nonzero_num / (pos - self.nextCheckPos + 1)) >= 0.95:
            self.nextCheckPos = pos

        self.used[begin] = True

        # base[begin] 记录 parent chr  -- 这样就可以从节点回溯得到字符串
        # 想要可以回溯的话，就不能在字符串末尾节点记录值了，或者给叶子节点找个0以外的值？ 0->-1
        # self.base[begin] = parent_base_idx     #【*】
        # print 'begin:',begin,self.base[begin]

        if self.size < begin + siblings[-1].code + 1:
            self.size = begin + siblings[-1].code + 1

        for sibling in siblings:  # 更新所有子节点的check     base[s]+c=t & check[t]=s
            self.check[begin + sibling.code] = begin

        for sibling in siblings:  # 由于是递归的情况，需要先处理完check
            # darts-java 还考虑到叶子节点有值的情况，暂时不考虑(需要记录的话，记录在叶子节点上)
            if sibling.code == -1:
                self.base[begin + sibling.code] = -1 * sibling.left - 1
            else:
                new_sibings = self.fetch(sibling)
                h = self.insert(new_sibings, begin + sibling.code)  # 插入孙子节点，begin + sibling.code为子节点的位置
                self.base[begin + sibling.code] = h  # 更新base所有子节点位置的转移基数为[其孩子最合适的begin]

        return begin

    def search(self, word):
        """ 查找单词是否存在 """
        p = 0  # root
        if word == '':
            return False
        for c in word:
            c = ord(c)
            next = abs(self.base[p]) + c
            # print(c, next, self.base[next], self.check[next])
            if next > self.MAX_SIZE:  # 一定不存在
                return False
            # print(self.base[self.base[p]])
            if self.check[next] != abs(self.base[p]):
                return False
            p = next

        # print('*'*10+'\n', 0, p, self.base[self.base[p]], self.check[self.base[p]])
        # 由于code=0,实际上是base[leaf_node->base+leaf_node.code]，这个负的值本身没什么用
        # 修正：left code = -1
        if self.base[self.base[p] - 1] < 0 and self.base[p] == self.check[self.base[p] - 1]:
            # print p
            return True
        else:  # 不是词尾
            return False

    def common_prefix_search(self, content):
        """ 公共前缀匹配 """
        # 用了 darts-java 写法，再仔细看一下
        result = []
        b = self.base[0]  # 从root开始
        p = 0
        n = 0
        tmp_str = ""
        for c in content:
            c = ord(c)
            p = b
            n = self.base[p - 1]  # for iden leaf

            if b == self.check[p - 1] and n < 0:
                result.append(tmp_str)

            tmp_str += chr(c)
            # print(tmp_str )
            p = b + c  # cur node

            if b == self.check[p]:
                b = self.base[p]  # next base
            else:  # no next node
                return result

        # 判断最后一个node
        p = b
        n = self.base[p - 1]

        if b == self.check[p - 1] and n < 0:
            result.append(tmp_str)

        return result

    def Find_Last_Base_index(self, word):
        b = self.base[0]  # 从root开始
        p = 0
        # n = 0
        # print len(word)
        tmp_str = ""
        for c in word:
            c = ord(c)
            p = b
            p = b + c  # cur node, p is new base position, b is the old

            if b == self.check[p]:
                tmp_str += chr(c)
                b = self.base[p]  # next base
            else:  # no next node
                return -1
        # print '====', p, self.base[p], tmp_str.decode('utf-8')
        return p

    def GetAllChildWord(self, index):
        result = []
        # result.append("")
        # print self.base[self.base[index]-1],'++++'
        if self.base[self.base[index] - 1] <= 0 and self.base[index] == self.check[self.base[index] - 1]:
            result.append("")
            # return result
        for i in range(0, 256):
            # print(chr(i))
            if self.check[self.base[index] + i] == self.base[index]:
                # print self.base[index],(chr(i)),i
                for s in self.GetAllChildWord(self.base[index] + i):
                    # print s
                    result.append(chr(i) + s)
        return result

    def FindAllWords(self, word):
        result = []
        last_index = self.Find_Last_Base_index(word)
        if last_index == -1:
            return result
        for end in self.GetAllChildWord(last_index):
            result.append(word + end)
        return result

    def get_string(self, chr_id):
        """ 从某个节点返回整个字符串, todo:改为私有 """
        if self.check[chr_id] == -1:
            raise Exception("不存在该字符。")
        child = chr_id
        s = []
        while 0 != child:
            base = self.check[child]
            print(base, child)
            label = chr(child - base)
            s.append(label)
            print(label)
            child = self.base[base]
        return "".join(s[::-1])

    def get_use_rate(self):
        """ 空间使用率 """
        return self.size / self.MAX_SIZE


if __name__ == '__main__':


    # for word in words:print [word]  #一个汉字的占用3个字符，
    words = []
    for line in open('/data/ylx/ylx/data/entities.txt').readlines():
        #    #print line.strip().decode('utf-8')
        words.append(line.strip())

    datrie = DATrie()
    datrie.build(words)
    # for line in open('1000.txt').readlines():
    #    print(datrie.search(line.strip()),end=' ')
    # print('-'*10)
    # print(datrie.search("景华路"))
    # print('-'*10)
    # print(datrie.search("景华路号"))

    # print('-'*10)
    # for item in datrie.common_prefix_search("商业模式"): print(item.decode('utf-8'))
    # for item in datrie.common_prefix_search("商业模式"):print item.decode('utf-8')
    # print(datrie.common_prefix_search("一举成名天下知"))
    # print(datrie.base[:1000])
    # print('-'*10)
    # print(datrie.get_string(21520))
    # index=datrie.Find_Last_Base_index("商业")
    # print(index),'-=-=-='
    # print datrie.search("商业"),datrie.search("商业"),datrie.search("商业模式")
    # print index, datrie.check[datrie.base[index]+230],datrie.base[index]
    for ii in datrie.FindAllWords('小红帽特工队的续作是？'):
        print (ii.decode('utf-8'))
    # print(datrie.Find_Last_Base_index("一举")[2].decode('utf-8'))
# print()