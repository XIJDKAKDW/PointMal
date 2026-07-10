#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：GetMultipleMetrix.py
@IDE     ：PyCharm
@Author  ：常晓松
@Date    ：2025/1/7 10:00
'''
import fnmatch
import json
import os
import pickle
import platform
import random
import shutil
import subprocess
import tempfile
import threading
from typing import Dict, List

import javalang
import networkx as nx
import numpy as np
import torch
from javalang.tree import MethodDeclaration, ConstructorDeclaration
from matplotlib import pyplot as plt
from networkx.algorithms import isomorphism
from torch.utils.data import Dataset


class SmaliDecompiler:
    def __init__(self, baksmali_path="smali-2.5.2.jar", jadx_path="jadx/bin/jadx"):
        self.baksmali_path = baksmali_path
        self.jadx_path = jadx_path

    def decompile_smali_to_java(self, smali_file_path, output_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 将smali文件重新打包为dex文件
            dex_file = os.path.join(temp_dir, "classes.dex")
            self._assemble_smali_to_dex(smali_file_path, dex_file)
            # 2. 使用jadx将dex反编译为Java
            self._decompile_dex_to_java(dex_file, output_dir)
            # print(f"反编译完成，结果保存在: {output_dir}")
            return True

    def decompile_smali_directory(self, smali_dir_path, output_dir):
        """
        将整个smali目录反编译为Java代码
        """
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. 将整个smali目录打包为dex文件
                dex_file = os.path.join(temp_dir, "classes.dex")
                self._assemble_smali_dir_to_dex(smali_dir_path, dex_file)

                # 2. 使用jadx将dex反编译为Java
                self._decompile_dex_to_java(dex_file, output_dir)

                # print(f"反编译完成，结果保存在: {output_dir}")
                return True

        except Exception as e:
            print(f"反编译失败: {e}")
            return False

    def _assemble_smali_to_dex(self, smali_file_path, output_dex_path):
        """使用smali工具将smali文件汇编为dex"""
        # 需要先将单个smali文件放到临时目录中
        with tempfile.TemporaryDirectory() as temp_smali_dir:
            # 复制smali文件到临时目录
            shutil.copy(smali_file_path, temp_smali_dir)
            self._assemble_smali_dir_to_dex(temp_smali_dir, output_dex_path)

    def _assemble_smali_dir_to_dex(self, smali_dir_path, output_dex_path):
        """使用smali工具将smali目录汇编为dex"""
        cmd = [
            "java", "-jar", self.baksmali_path,
            "assemble", smali_dir_path,
            "-o", output_dex_path

        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or len(result.stderr)>0:
            raise Exception(f"smali汇编失败: {result.stderr}")

    def _decompile_dex_to_java(self, dex_file_path, output_dir):
        """使用jadx将dex文件反编译为Java"""
        cmd = [
            self.jadx_path,
            "-d", output_dir,
            "--no-imports",  # 不优化imports
            "--no-res",      # 不处理资源
            "--decompilation-mode", "restructure",  # 改为auto模式（或其他可用模式）
            "--show-bad-code",  # 显示有问题的代码
            dex_file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or len(result.stderr)>0 or len(result.stderr)>0:
            if len(result.stderr)==0:
                raise Exception("jadx反编译失败:"+result.stdout)
            else:
                raise Exception("jadx反编译失败:"+result.stderr)

class JavaMethodExtractor:
    def extract_methods_from_java_content(self, java_content, class_name):
        """
        从Java代码内容中提取所有方法信息
        返回格式: {方法签名: 方法信息}
        """
        methods_dict = {}
        try:

            #替换 不支持的类声明
            pattern = r'new\s+(\d+)\('
            replacement = r'new a\1('
            java_content=re.sub(pattern, replacement, java_content)
            tree = javalang.parse.parse(java_content)

            for path, node in tree:
                if isinstance(node, MethodDeclaration):
                    method_signature = self._get_method_signature(node, class_name)
                    method_full_text = self._extract_full_method_text(java_content, node)

                    methods_dict[method_signature] = {
                        'name': node.name,
                        'return_type': str(node.return_type) if node.return_type else 'void',
                        'parameters': [self._format_parameter(param) for param in node.parameters],
                        'modifiers': list(node.modifiers) if hasattr(node, 'modifiers') else [],
                        'body': self._extract_method_body(java_content, node),
                        'full_method_text': method_full_text,  # 添加完整的函数文本
                        'class': class_name
                    }
                elif isinstance(node, ConstructorDeclaration):
                    method_signature = self._get_constructor_signature(node, class_name)
                    method_full_text = self._extract_full_method_text(java_content, node)

                    methods_dict[method_signature] = {
                        'name': node.name,
                        'return_type': 'void',  # 构造函数没有返回类型
                        'parameters': [self._format_parameter(param) for param in node.parameters],
                        'modifiers': list(node.modifiers) if hasattr(node, 'modifiers') else [],
                        'body': self._extract_method_body(java_content, node),
                        'full_method_text': method_full_text,
                        'class': class_name,
                        'type': 'constructor'
                    }
        except Exception as e:
            methods_dict = self._fallback_extract_methods(java_content, class_name)

            print(f"解析Java内容失败: {e}")
            import traceback
            traceback.print_exc()

        return methods_dict

    def _fallback_extract_methods(self, java_content, class_name):
        """
        备选方法：使用正则表达式提取方法（当语法解析失败时使用）
        """
        methods_dict = {}

        # 匹配方法（包括构造函数）的正则表达式
        method_pattern = r'((?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)*([\w\<\>\[\]]+\s+)?(\w+)\s*\(([^)]*)\)\s*\{[^}]*\}'
        constructor_pattern = r'(public|private|protected)\s+(\w+)\s*\(([^)]*)\)\s*\{[^}]*\}'

        # 查找所有方法
        methods = re.finditer(method_pattern, java_content, re.MULTILINE | re.DOTALL)

        for match in methods:
            modifiers = match.group(1).strip().split() if match.group(1) else []
            return_type = match.group(2).strip() if match.group(2) else 'void'
            method_name = match.group(3).strip()
            parameters = match.group(4).strip() if match.group(4) else ''

            # 跳过明显不是方法的情况（如if、for等）
            if method_name in ['if', 'for', 'while', 'switch', 'catch']:
                continue

            # 判断是否是构造函数
            is_constructor = (return_type == '' and method_name == class_name.split('.')[-1])

            if is_constructor:
                method_signature = f"{class_name}.{method_name}({parameters})"
                method_type = 'constructor'
                return_type = 'void'
            else:
                method_signature = f"{class_name}.{method_name}({parameters})"
                method_type = 'method'

            methods_dict[method_signature] = {
                'name': method_name,
                'return_type': return_type,
                'parameters': [p.strip() for p in parameters.split(',')] if parameters else [],
                'modifiers': modifiers,
                'body': match.group(0),
                'full_method_text': match.group(0),
                'class': class_name,
                'type': method_type
            }

        return methods_dict
    def _get_constructor_signature(self, constructor_node, class_name):
        """生成构造函数的签名"""
        params = []
        for param in constructor_node.parameters:
            param_type = str(param.type)
            if param.type.dimensions:
                param_type += '[]' * len(param.type.dimensions)
            params.append(f"{param_type} {param.name}")

        return f"{class_name}.{constructor_node.name}({', '.join(params)})"
    def _get_method_signature(self, method_node, class_name):
        """生成方法签名"""
        return_type = str(method_node.return_type) if method_node.return_type else 'void'
        params = ', '.join([str(param.type) for param in method_node.parameters])
        return f"{class_name}.{method_node.name}({params}):{return_type}"

    def _format_parameter(self, param):
        """格式化参数信息"""
        return {
            'name': param.name,
            'type': str(param.type),
            'modifiers': getattr(param, 'modifiers', [])
        }

    def _extract_method_body(self, java_content, method_node):
        """提取方法的完整函数体"""
        lines = java_content.split('\n')
        start_line = method_node.position.line - 1 if method_node.position else 0

        # 找到方法体的开始和结束
        brace_count = 0
        in_body = False
        body_lines = []

        for i in range(start_line, len(lines)):
            line = lines[i].strip()

            if not in_body and '{' in line:
                in_body = True
                brace_count += line.count('{')
                continue

            if in_body:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                body_lines.append(lines[i])

                if brace_count <= 0:
                    break

        return '\n'.join(body_lines)

    def _extract_full_method_text(self, java_content, method_node):
        """
        提取完整的函数定义文本，包括修饰符、返回类型、方法名、参数和方法体
        """
        lines = java_content.split('\n')
        start_line = method_node.position.line - 1 if method_node.position else 0

        # 找到方法声明的开始行
        method_start_line = start_line
        while method_start_line > 0:
            prev_line = lines[method_start_line - 1].strip()
            # 如果上一行是空行或注释，继续向上查找
            if not prev_line or prev_line.startswith('//') or prev_line.startswith('/*') or prev_line.startswith('*'):
                method_start_line -= 1
            else:
                break

        # 找到方法体的结束
        brace_count = 0
        in_method = False
        method_lines = []

        for i in range(method_start_line, len(lines)):
            line = lines[i]

            # 检查是否进入方法体
            if not in_method and '{' in line:
                in_method = True
                brace_count += line.count('{')
            elif in_method:
                brace_count += line.count('{')
                brace_count -= line.count('}')

            method_lines.append(line)

            if in_method and brace_count <= 0:
                break

        return '\n'.join(method_lines)

def get_attn_pad_mask(seq_q):
    batch_size,channel,_=seq_q.size()
    pad_attn_mask=torch.randint(1, 5, (batch_size, channel,channel))
    pad_attn_mask=pad_attn_mask.not_equal(10)
    # pad_attn_mask=pad_attn_mask.expand(batch_size,len_q,len_k)
    return pad_attn_mask



class ThreadOwn(threading.Thread):
    def __init__(self, func, args=()):
        super(ThreadOwn, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None
def propagate_info(graph):
    """
    遍历图并进行双向信息传播
    """
    # 子节点信息传播到父节点
    for node in graph.nodes():
        # 获取当前节点的所有父节点
        parents = list(graph.predecessors(node))

        for parent in parents:
            # 子节点 -> 父节点传播
            parent.permission.extend(node.permission)
            parent.sensitiveApi.extend(node.sensitiveApi)
            parent.suspiciousApi.extend(node.suspiciousApi)
            parent.url.extend(node.url)

    # 父节点信息传播到子节点
    for node in graph.nodes():
        # 获取当前节点的所有子节点
        children = list(graph.successors(node))

        for child in children:
            # 父节点 -> 子节点传播
            child.filterToken.extend(node.filterToken)
            child.provider.extend(node.provider)
            child.hardware_component.extend(node.hardware_component)
            if node.component:
                child.component = node.component

class Node:
    def __init__(self,inheritance_tree, name=''):
        self.name = name  #类名+方法名
        self.super=None
        if inheritance_tree is not None:
            self.super =inheritance_tree.get_direct_parent_class(self.name.split('->')[0])  #父类
        self.body=''    		#函数内容
        self.permission=[]      # 权限
        self.sensitiveApi=[]    #敏感API
        self.suspiciousApi=[]   #可疑api

        self.url=[]             	#url
        self.filterToken=[]         #过滤器标识
        self.provider=[]            #内容
        self.hardware_component=[]  #硬件组件
        self.component=''		    #活动、服务、广播、内容

        # 新增：存储反编译后的Java代码
        self.java_code = ''

        self.AndroidSuspiciousApis = ["getExternalStorageDirectory", "getSimCountryIso", "execHttpRequest", "sendTextMessage", "getSubscriberId", "getDeviceId", "getPackageInfo", "getSystemService", "getWifiState","setWifiEnabled", "setWifiDisabled", "Cipher"]
        self.OtherSuspiciousApis = ["Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)",
                                    "Ljava/net/HttpURLconnection",
                                    "Lorg/apache/http/client/methods/HttpPost",
                                    "Landroid/telephony/SmsMessage;->getMessageBody",
                                    "Ljava/io/IOException;->printStackTrace",
                                    "Ljava/lang/Runtime;->exec"]
        self.NotLikeApis = ["system/bin/su", "android/os/Exec"]



    def setComponent(self,component):
        self.component=component
    def setFilterToken(self,filterToken):
        self.filterToken=filterToken
    def setProvider(self,provider):
        self.provider=provider
    def setHardwareComponent(self,hardware_component):
        self.hardware_component=hardware_component


    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        return False

    def __repr__(self):
        return self.name
    def addBody(self,content):
        self.body+=(content+'\r\n')
    def setName(self,caller_method):
        self.name=caller_method
        return self
    def setUrl(self,context):
        URLSearch = re.search("https?://([\da-z\.-]+\.[a-z\.]{2, 6}|[\d.]+)[^'\"]*", context, re.IGNORECASE)
        if (URLSearch):
            URL = URLSearch.group()
            Domain = re.sub("https?://(.*)", "\g<1>",
                            re.search("https?://([^/:\\\\]*)", URL, re.IGNORECASE).group(), 0, re.IGNORECASE)
            self.url.append(Domain)
    def setSensitiveApis(self,PermApiDictFromJson,callee_method):
        if ";->" in callee_method:
            ApiParts = callee_method.split(";->")
            ApiClass = ApiParts[0].strip()
            ApiName = ApiParts[1].split("(")[0].strip()
            ApiDetails = {}
            ApiDetails['FullApi'] = callee_method
            ApiDetails['ApiClass'] = ApiClass
            ApiDetails['ApiName'] = ApiName
            Permission = GetPermFromApi(ApiClass,
                                        ApiDetails['ApiName'],
                                        PermApiDictFromJson)
            if(Permission != None):
                self.permission.append(Permission)
                api=ApiDetails['ApiClass'].replace("/", ".").replace("Landroid", "android").strip()+'.'+ApiDetails['ApiName']
                self.sensitiveApi.append(api)

    def setSuspeciousApis(self,context):
        #可疑api
        Parts = context.split(",")
        for Part in Parts:
            if ";->" in Part:
                Part = Part.strip()
                if Part.startswith('Landroid'):
                    FullApi = Part
                    ApiParts = FullApi.split(";->")
                    ApiClass = ApiParts[0].strip()
                    ApiName = ApiParts[1].split("(")[0].strip()
                    if(ApiName in self.AndroidSuspiciousApis):
                        self.suspiciousApi.append(ApiClass+"."+ApiName)
            for Element in self.OtherSuspiciousApis:
                if(Element in Part):
                    self.suspiciousApi.append(Element)
        for Element in self.NotLikeApis:
            if Element in context:
                self.suspiciousApi.append(Element)

def getShortestSequencesToLeaves(nx_graph: nx.Graph, root_node: Node, leaf_nodes: List[Node]) -> Dict[Node, List[Node]]:

    if root_node not in nx_graph:
        return {}

    # BFS找到所有节点的最短路径
    queue = deque([root_node])
    parent = {root_node: None}

    while queue:
        current_node = queue.popleft()

        # 获取当前节点的所有邻居
        neighbors = list(nx_graph.neighbors(current_node))

        for neighbor in neighbors:
            if neighbor not in parent:
                parent[neighbor] = current_node
                queue.append(neighbor)

    # 为每个叶子节点构建最短路径
    sequences = {}
    for leaf in leaf_nodes:
        if leaf in parent:
            # 回溯到根节点
            path = []
            current = leaf
            while current is not None:
                path.append(current)
                current = parent.get(current)
            sequences[leaf] = list(reversed(path))

    return sequences

def get_sequeces(G,root,leaves):
    # 获取最短序列集合
    sequences = getShortestSequencesToLeaves(G, root, leaves)
    called_methods={}
    for leaf, path in sequences.items():
        if len(path)>1:
            called_methods[path[1]]=[node  for node in path if node !=path[0] and node !=path[1]]
    return called_methods

import re

def extract_function_name(s):
    # 匹配 -> 后面到 ( 之前的部分
    match = re.search(r'->([^(]+)\(', s)
    return ' '+match.group(1)+'(' if match else None


def is_android_base_package(smali_file):
    base_packages = {
        "Landroid/",
        "Landroidx/",
        "Lcom/android/",
        "Ljava/",
        "Ljavax/",
        "Lsun/",
        "Lorg/xml/",
        "Lorg/json/",
        "Lorg/w3c/",
        "Lorg/apache/harmony/",
        "Lcom/google/android/"
    }

    with open(smali_file, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            # 检查类定义行
            if line.startswith('.class'):
                # 提取完整的类名（如：Lcom/androidquery/auth/AccountHandle;）
                class_name = line.split(' ')[-1]
                # 检查是否为基础包
                for base_pkg in base_packages:
                    if class_name.startswith(base_pkg):
                        return True
                return False

    return False

# 将调用关系加到图
graph_lock = threading.Lock()
def save_call(smali_file, graph, PermApiDictFromJson, apkAnalyzer, smaliAnalyzer=None, inheritance_tree=None):
    try:
        f = open(smali_file, 'r', encoding='UTF-8')
        caller_class = ''
        caller_method = ''
        current_node = None
        current_component=''
        with graph_lock:
            node_dict = {node.name: node for node in graph.nodes()}

        methodBody=False
        for line in f:
            if len(line)==0:
                continue
            line = line.strip().replace("\n", "")

            line_list = line.strip().split(' ')
            #url
            if current_node!=None:
                current_node.setUrl(line)
            # 找到类名
            if line.startswith(".class") and len(line_list)> 1:
                caller_class = line_list[len(line_list) - 1]
                # 找到父类名
                parent_classes = inheritance_tree.get_all_parent_classes(caller_class)
                if len(parent_classes)>0:
                    for one in parent_classes:
                        if one =='Landroid/app/Activity;':
                            current_component='Activity'
                        if one =='Landroid/app/Service;':
                            current_component='Service'
                        if one =='Landroid/content/BroadcastReceiver;':
                            current_component='BroadcastReceiver'
                        if one =='Landroid/content/ContentProvider;':
                            current_component='ContentProvider'
                        if one =='Landroid/app/Application;':
                            current_component='Application'

            # 找到函数
            elif line.startswith(".method") and len(line_list)> 1:
                methodBody=True
                caller_method = caller_class + "->" + line_list[len(line_list) - 1]
                with graph_lock:
                    if caller_method in node_dict:
                        current_node = node_dict[caller_method]
                    else:
                        current_node = Node(inheritance_tree,caller_method)
                        if len(current_component)>0:
                            current_node.setComponent(current_component)
                            actions = apkAnalyzer.search_actions_by_classname(caller_method)
                            if len(actions)>0:
                                current_node.setFilterToken(actions)
                        graph.add_node(current_node)
                        node_dict[caller_method] = current_node  # 更新字典
            # 找到调用类
            elif line.startswith("invoke-") and len(line_list) > 1:
                callee_method = line_list[len(line_list) - 1]
                #可疑api
                current_node.setSuspeciousApis(line)
                if caller_method != '':
                    if smaliAnalyzer._is_project_invoke(line):
                        with graph_lock:
                            if callee_method in node_dict:
                                callee_node = node_dict[callee_method]
                            else:
                                callee_node = Node(inheritance_tree,callee_method)
                                graph.add_node(callee_node)
                                node_dict[callee_method] = callee_node
                            if not graph.has_edge(current_node, callee_node):
                                graph.add_edge(current_node, callee_node)
                    else:
                        #获取敏感api、权限
                        current_node.setSensitiveApis(PermApiDictFromJson,callee_method)
            # 终止方法
            elif line.startswith(".end method"):
                caller_method = ''
                current_node.addBody(line)
                methodBody=False
            if methodBody:
                current_node.addBody(line)
        f.close()
    except FileNotFoundError:
        pass

def GetPermFromApi(ApiClass, ApiMethodName,PermApiDictFromJson):
    ApiClass=ApiClass.replace("/", ".").replace("Landroid", "android").strip()
    ApiClass=ApiClass.lower()
    ApiMethodName=ApiMethodName.lower()
    ApiName=ApiClass+ApiMethodName
    if(ApiClass+ApiMethodName) in PermApiDictFromJson:
        return PermApiDictFromJson[ApiName]
    else:
        return None


class SmaliInheritanceTree:
    def __init__(self):
        self.class_hierarchy = {}
        self.class_files = {}

    def parse_smali_file(self, file_path):
        """解析单个smali文件，提取类名和父类信息"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用正则表达式匹配类定义和父类
            class_match = re.search(r'\.class\s+(?:\w+\s+)*L([^;]+);', content)
            super_match = re.search(r'\.super\s+L([^;]+);', content)

            if class_match and super_match:
                class_name = 'L' + class_match.group(1) + ';'
                super_class = 'L' + super_match.group(1) + ';'

                # 存储类文件路径
                self.class_files[class_name] = file_path

                # 构建继承关系
                self.class_hierarchy[class_name] = super_class

                return class_name, super_class

        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            return None, None

    def build_inheritance_tree(self, smali_files):

        # 首先解析所有文件
        for file_path in smali_files:
            if os.path.exists(file_path):
                self.parse_smali_file(file_path)
            else:
                print(f"文件不存在: {file_path}")

        return self.class_hierarchy

    def get_all_parent_classes(self, class_name):
        """获取指定类的所有上级父类"""
        if class_name not in self.class_hierarchy:
            return []

        parents = []
        current_class = class_name

        # 沿着继承链向上查找
        max_depth = 50  # 防止无限循环
        depth = 0

        while current_class in self.class_hierarchy and depth < max_depth:
            parent_class = self.class_hierarchy[current_class]
            parents.append(parent_class)
            current_class = parent_class
            depth += 1

            # Java所有类的最终父类是java/lang/Object
            if parent_class == 'Ljava/lang/Object;':
                break

        return parents

    def get_direct_parent_class(self, class_name):
        """获取指定类的直接父类"""
        if class_name not in self.class_hierarchy:
            return None

        # 直接从class_hierarchy中获取父类
        parent_class = self.class_hierarchy.get(class_name)
        return parent_class
    def find_class_file(self, class_name):
        """查找类对应的文件路径"""
        return self.class_files.get(class_name, "未找到")

def save_call_multi(one_group, graph, PermApiDictFromJson, apkAnalyzer, smaliAnalyzer=None, inheritance_tree=None):
    for smali_file in one_group:
        save_call(smali_file, graph, PermApiDictFromJson, apkAnalyzer, smaliAnalyzer, inheritance_tree)
def gen_call_graph_multi(smali_loc, PermApiDictFromJson, apkAnalyzer, smaliAnalyzer=None,thread_num=3):
    graph = nx.DiGraph()
    smali_list=[]
    smali_group_list=[]
    if not os.path.exists(smali_loc):
        return None
    all_decode_file = os.listdir(smali_loc)
    for f in all_decode_file:
        path = smali_loc + os.sep+f
        for dirpath, dirs, files in os.walk(path):
            for filename in fnmatch.filter(files, '*.smali'):
                smali_list.append(dirpath + os.sep+ filename)
    #继承树
    inheritance_tree = SmaliInheritanceTree()
    inheritance_tree.build_inheritance_tree(smali_list)


    group_len=int(len(smali_list)/thread_num)+ (1 if len(smali_list)%thread_num>0 else 0)
    for i in range(thread_num):
        smali_group_list.append(smali_list[group_len*i:group_len*(i+1)])
    threads=[]
    for one_group in smali_group_list:
        threads.append(ThreadOwn(save_call_multi, (one_group, graph, PermApiDictFromJson, apkAnalyzer, smaliAnalyzer, inheritance_tree)))
    for thread_tmp in threads:
        thread_tmp.start()
    for thread_tmp in threads:
        thread_tmp.join(60*5)

    #增加项目层的特征入图
    current_node = Node(inheritance_tree,'apk-base')
    if len(apkAnalyzer.provider_names)>0:
        current_node.setProvider(apkAnalyzer.provider_names)
    if len(apkAnalyzer.hardware_component)>0:
        current_node.setHardwareComponent(apkAnalyzer.hardware_component)
    filter_token = apkAnalyzer.get_all_filter_actions()
    permission = apkAnalyzer.get_uses_permissions()
    for node in graph.nodes:
        if len(node.filterToken)>0:
            filter_token = [item for item in filter_token if item not in node.filterToken]
        if len(node.permission)>0:
            permission = [item for item in permission if item not in node.permission]

    if len(filter_token)>0:
        current_node.setFilterToken(filter_token)
    if len(permission)>0:
        current_node.permission=permission
    graph.add_node(current_node)
    return graph

def generate_apk_method_graph(smali_loc, PermApiDictFromJson, apkAnalyzer, smaliAnalyzer=None,thread_num=3):
    # 生成函数调用图
    sfcg = gen_call_graph_multi(smali_loc, PermApiDictFromJson, apkAnalyzer, smaliAnalyzer,thread_num)
    return sfcg
def decompile_all_nodes(graph, smali_dir, thread_num=1):
    """
    对图中所有节点进行批量反编译（按包分组），并行处理
    """
    jadx_path = r"D:\jadx\bin\jadx.bat"  # 根据实际情况调整路径
    smali_path = r"D:\\dexCompile\\program\\smali-2.5.2.jar"
    system = platform.system()
    if system == "Linux":
        jadx_path = r"/home/changxiaosong/jadx/bin/jadx"
        smali_path = r'/home/changxiaosong/python/malwareTest/smali-2.5.2.jar'

    decompiler = SmaliDecompiler(smali_path, jadx_path)

    # 按类名分组节点，并重构为完整的类结构
    class_structure = _build_class_structure(graph)

    # 将类分组以便并行处理
    class_items = list(class_structure.items())
    class_groups = []
    group_size = max(1, len(class_items) // thread_num)

    for i in range(0, len(class_items), group_size):
        class_groups.append(class_items[i:i + group_size])

    # 创建并启动线程
    threads = []
    for group in class_groups:
        thread = ThreadOwn(decompile_class_group, (group, decompiler))
        thread.start()
        threads.append(thread)

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print(f"批量反编译完成，共处理 {len(class_structure)} 个类")

def _build_class_structure(graph):
    """
    构建完整的类结构，将相同类的方法合并
    """
    class_structure = {}

    for node in graph.nodes():
        if hasattr(node, 'smali_code') and node.smali_code:
            # 提取类名（从方法名中提取类部分）
            if '->' in node.name:
                class_name = node.name.split('->')[0]
                method_name = node.name.split('->')[1]
            else:
                class_name = node.name
                method_name = "class"

            if class_name not in class_structure:
                # 初始化类结构
                class_structure[class_name] = {
                    'super': getattr(node, 'super', 'Ljava/lang/Object;'),
                    'methods': {},
                    'class_node': node  # 保存一个节点用于获取类级别信息
                }

            # 存储方法信息
            class_structure[class_name]['methods'][method_name] = {
                'node': node,
                'smali_code': node.smali_code
            }

    return class_structure

def decompile_class_group(class_group, decompiler):
    """
    并行处理一组类
    """
    for class_name, class_info in class_group:
        try:
            _decompile_single_class(class_name, class_info, decompiler)
        except Exception as e:
            print(f"批量反编译类 {class_name} 失败: {e}")
            # 为失败的节点设置错误信息
            for method_name, method_info in class_info['methods'].items():
                method_info['node'].java_code = f"// 批量反编译失败: {str(e)}"

def _decompile_single_class(class_name, class_info, decompiler):
    """
    处理单个类的反编译，将整个类作为一个smali文件处理
    """
    # 创建临时目录用于存放该类的smali文件
    with tempfile.TemporaryDirectory() as temp_smali_dir:
        # 生成完整的类smali文件
        complete_smali = _generate_complete_smali_class(class_name, class_info)

        # 保存完整的smali文件
        smali_filename = f"{class_name.replace('/', '$').replace(';', '')}.smali"
        smali_file_path = os.path.join(temp_smali_dir, smali_filename)

        with open(smali_file_path, 'w', encoding='utf-8') as f:
            f.write(complete_smali)

        # 创建临时输出目录
        temp_output_dir = tempfile.mkdtemp()

        try:
            # 批量反编译整个目录
            success = decompiler.decompile_smali_directory(temp_smali_dir, temp_output_dir)

            if success:
                # 读取反编译结果并分配到对应节点
                java_files_content = read_java_files_by_class(temp_output_dir)

                # 将Java代码分配到对应方法节点
                _distribute_java_code_to_methods(class_name, class_info, java_files_content)
        finally:
            # 确保清理临时输出目录
            import shutil
            shutil.rmtree(temp_output_dir, ignore_errors=True)

def _generate_complete_smali_class(class_name, class_info):
    """
    生成完整的类smali代码，合并所有方法
    """
    # 从第一个方法节点获取类的基本信息
    first_method = next(iter(class_info['methods'].values()))
    first_node = first_method['node']

    # 构建类头
    smali_content = []
    smali_content.append(f".class public {class_name}")
    smali_content.append(f".super {class_info.get('super', 'Ljava/lang/Object;')}")
    smali_content.append("")

    # 收集所有方法体
    method_bodies = []

    for method_name, method_info in class_info['methods'].items():
        method_smali = method_info['smali_code']

        # 提取方法部分（从.method到.end method）
        method_lines = []
        in_method = False
        brace_count = 0

        for line in method_smali.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('.method'):
                in_method = True
                brace_count = 0
                method_lines = [line]
            elif line.startswith('.end method'):
                method_lines.append(line)
                method_bodies.append('\n'.join(method_lines))
                in_method = False
            elif in_method:
                method_lines.append(line)
                # 简单的大括号计数（用于处理嵌套情况）
                if '{' in line:
                    brace_count += line.count('{')
                if '}' in line:
                    brace_count -= line.count('}')

    # 添加所有方法体
    for method_body in method_bodies:
        smali_content.append(method_body)
        smali_content.append("")

    return '\n'.join(smali_content)
def _distribute_java_code_to_methods(class_name, class_info, java_files_content):
    """
    将反编译的Java代码分配到各个方法节点
    """
    # 将smali类名转换为Java类名格式
    java_class_name = class_name.replace('/', '.').replace('L', '').replace(';', '')
    simple_java_class_name = java_class_name.split('.')[-1] if '.' in java_class_name else java_class_name

    # 查找对应的Java类内容
    java_content = None
    for class_key, content in java_files_content.items():
        if simple_java_class_name in class_key or class_key in simple_java_class_name:
            java_content = content
            break

    if not java_content:
        error_msg = f"// 未找到对应的Java类: {java_class_name}"
        for method_name, method_info in class_info['methods'].items():
            method_info['node'].java_code = error_msg
        return

    # 使用JavaMethodExtractor提取所有方法
    extractor = JavaMethodExtractor()
    methods_dict = extractor.extract_methods_from_java_content(java_content, java_class_name)

    # 将Java方法分配到对应的节点
    for method_name, method_info in class_info['methods'].items():
        node = method_info['node']

        # 从smali方法名提取简单方法名
        if '(' in method_name:
            simple_method_name = method_name.split('(')[0]
        else:
            simple_method_name = method_name

        # 特殊处理构造函数：smali中的<init>对应Java中的类名
        if simple_method_name == '<init>':
            simple_method_name = simple_java_class_name

        # 查找匹配的Java方法
        found = False
        for java_method_signature, java_method_info in methods_dict.items():
            # 匹配方法名或构造函数
            if (simple_method_name in java_method_signature or
                    (simple_method_name == simple_java_class_name and
                     java_method_info.get('type') == 'constructor')):
                node.java_code = java_method_info.get('full_method_text',
                                                      java_method_info.get('body', '// 方法体未找到'))
                found = True
                break

        if not found:
            node.java_code = f"// 未找到对应的方法: {simple_method_name}"
def read_java_files_by_class(java_dir):
    """
    读取Java目录中的所有Java文件内容，按类名组织
    """
    java_files = {}
    for root, dirs, files in os.walk(java_dir):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 提取类名作为键
                        class_match = re.search(r'class\s+(\w+)', content)
                        if class_match:
                            class_name = class_match.group(1)
                            java_files[class_name] = content
                        else:
                            # 如果没有找到类名，使用文件名
                            java_files[file[:-5]] = content
                except Exception as e:
                    print(f"读取Java文件 {file_path} 失败: {e}")
    return java_files
def read_java_files(java_dir):
    """
    读取Java目录中的所有Java文件内容
    """
    java_content = ""
    for root, dirs, files in os.walk(java_dir):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        java_content += f.read() + "\n\n"
                except Exception as e:
                    print(f"读取Java文件 {file_path} 失败: {e}")
    return java_content
def get_graph_from_path(path_tmp):
    try:
        return nx.read_gexf(path_tmp)
    except:
        print('子图解析异常',path_tmp)
        return nx.DiGraph()
def get_api_set(api_path, threshold):
    api_set = set()
    f = open(api_path, "r")
    for line in f:
        line = line.strip().replace("\n", "")
        api_set.add(line)
    choosed_api_len = int(threshold * len(api_set))
    choosed_api_set = set(random.sample(api_set, choosed_api_len))
    f.close()
    return choosed_api_set
class Apk:
    def __init__(self, apk_name, apk_type, threshold, save_dir,baseDir):
        # apk的基本信息统计
        self.apk_name = apk_name
        self.apk_type = apk_type
        self.node_number = 0
        self.edge_number = 0
        self.fcg_graph_time = 0
        self.pattern_find_time = 0
        self.save_dir = save_dir
        self.dapasa_api_set = set()
        self.pscout_api_set = set()

        # 存储子图的总数，键值对为:子图--数目
        self.subgraph_dict= dict()

        # 存储子图的存在磁盘上的路径名称，键值对为:id--子图
        # 可以存储到文件，方便后期的信息统计
        self.subgraph_id_dict = dict()

        # 用于比对的敏感API部分
        self.sen_dapasa_api_set = get_api_set(baseDir+"sensitiveApiFromDAPASA.txt", threshold)
        self.sen_pscout_api_set = get_api_set(baseDir+"sensitiveApiFromPscout.txt", threshold)
def search_subgraph_pattern(apk, graph, node_num):
    dapasa_api_set = apk.dapasa_api_set
    pscout_api_set = apk.pscout_api_set
    sen_api_len = len(dapasa_api_set) + len(pscout_api_set)
    if sen_api_len > 500:
        neigh_search_len = 2
    elif sen_api_len > 50:
        if node_num < 6:
            neigh_search_len = 4
        else:
            neigh_search_len = 2
    else:
        if node_num < 6:
            neigh_search_len = 5
        else:
            neigh_search_len = 3
    if node_num == 3:
        neigh_search_len = 8

    get_subgraph_use_api(apk, graph, neigh_search_len, node_num, dapasa_api_set)
    get_subgraph_use_api(apk, graph, neigh_search_len, node_num, pscout_api_set)
#多线程重构
def get_subgraph_use_api(apk, graph, neigh_search_len, node_num, pscout_api_set):
    threads=[]
    for pscout_api in pscout_api_set:
        threads.append(ThreadOwn(get_subgraph_use_api_one, (apk, graph, neigh_search_len, node_num, pscout_api)))
    for thread_tmp in threads:
        thread_tmp.start()
    for thread_tmp in threads:
        thread_tmp.join()
def get_subgraph_use_api_one(apk, graph, neigh_search_len, node_num, pscout_api):
    subgraph_node_set = set()
    subgraph_node_set.add(pscout_api)
    dfs(pscout_api, graph, node_num, subgraph_node_set, neigh_search_len, apk)
    subgraph_node_set.remove(pscout_api)

# dfs搜索node_num深的图
def dfs(dapasa_api, fcg, node_num, subgraph_node_set, neigh_search_len, apk):
    # 子图节点数量大于规定子图节点数量，返回0,表示超过节点阈值，不找了
    if len(subgraph_node_set) > node_num:
        return
    # 子图节点数量等于规定子图节点数量，打印子图，
    if len(subgraph_node_set) == node_num:
        subgraph = make_graph(fcg, subgraph_node_set)
        judge_is_Iso(subgraph, apk)
        return
    # 子图数量小于规定子图节点数量，搜索。。
    neigh_list = list(fcg.successors(dapasa_api))
    neigh_list.extend(list(fcg.predecessors(dapasa_api)))
    neigh_searched_len = 0
    for neigh_node in neigh_list:
        # 邻居访问过，直接跳过
        if neigh_node in subgraph_node_set:
            continue
        # 访问当前节点，并继续dfs
        subgraph_node_set.add(neigh_node)
        dfs(neigh_node, fcg, node_num, subgraph_node_set, neigh_search_len, apk)
        subgraph_node_set.remove(neigh_node)
        neigh_searched_len += 1
        # 超过节点的访问范围，自动退出
        if neigh_searched_len > neigh_search_len:
            break
def make_graph(fcg, adj):
    H = nx.DiGraph()
    if len(adj) == 1:
        H.add_node(adj[0])
        return H
    for i in adj:
        for j in adj:
            if i == j:
                continue
            if not H.has_node(i):
                H.add_node(i)
            if not H.has_node(j):
                H.add_node(j)
            if fcg.has_edge(i, j):
                H.add_edge(i, j)
    return H
def judge_is_Iso(subgraph, apk):
    if not apk.subgraph_dict:
        apk.subgraph_dict[subgraph] = 1
        #apk.subgraph_id_dict[len(apk.subgraph_dict)] = subgraph
        nx.write_gexf(subgraph, os.path.join(apk.save_dir, str(len(apk.subgraph_dict))))
        return False
    for pattern in list(apk.subgraph_dict.keys()):
        matcher = isomorphism.DiGraphMatcher(subgraph, pattern)
        # if isomorphism add number 1
        if matcher.is_isomorphic():
            apk.subgraph_dict[pattern] = apk.subgraph_dict[pattern] + 1
            return True
    apk.subgraph_dict[subgraph] = 1
    nx.write_gexf(subgraph, os.path.join(apk.save_dir, str(len(apk.subgraph_dict))))
    return False
def get_all_child(graph,one_node,child_all,deep):
    deep=deep-1
    if deep>0:
        neigh_list = list(graph.successors(one_node))
        if len(neigh_list)==0:
            return
        else:
            for one in neigh_list:
                if one not in child_all:
                    child_all.append(one)
                    get_all_child(graph,one,child_all,deep)
def grap_matrix_3d(list,name):
    baseDir=r'C:\Users\chang\Desktop'
    ax = plt.axes(projection='3d')
    x=0
    for one_pic in list:
        y=0
        for row in one_pic:
            z=0
            for point in row:
                if point>=1:
                    color='r'
                    ax.scatter3D(x,y,z,c=color,s=10)
                # else:
                #     color='b'
                #     ax.scatter3D(x,y,z,c=color,s=10,alpha=0.2)
                z+=1
            y+=1
        x+=1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.savefig(baseDir+os.sep+str(name)+".png")
    # plt.show()
def get_filelist(dir, Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)
    # # 若只是要返回文件文，使用这个
    # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):

        for child in os.listdir(dir):
            get_filelist(dir+os.sep+child,Filelist)

class ApkDataset(Dataset):
    def __init__(self, feature_file, label_file):
        with open(feature_file, 'rb') as file:
            features = pickle.load(file)
        with open(label_file, 'rb') as file:
            labels = pickle.load(file)
        labels=[[0] if i=='B' else [1] for i in labels]
        labels=np.array(labels)
        # labels = torch.FloatTensor(labels)
        features=np.array(features,dtype=np.float32)
        # features=torch.from_numpy(features)

        self.X_data = features
        self.Y_data = labels
    def __init__(self, features, labels,type='list'):
        labels=[[0] if i=='B' else [1] for i in labels]
        labels=np.array(labels)
        # labels = torch.FloatTensor(labels)
        features=np.array(features,dtype=np.float32)
        # features=torch.from_numpy(features)

        self.X_data = features
        self.Y_data = labels

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.from_numpy(self.X_data[idx])  # 转换为 Tensor
        y = torch.FloatTensor(self.Y_data[idx])
        return x, y


def get_sensitive_apis_extend(sensitive_api_list):
    sensitive_api_files = []
    for dirpath, dirs, files in os.walk(sensitive_api_list):
        for one in files:
            sensitive_api_files.append(dirpath + os.sep + one)
    PermApiDictFromJson = {}
    for one in sensitive_api_files:
        with open(one, 'rb') as FH:
            # Use SmallCase json file to prevent run time case conversion in GetPermFromApi
            tmp = json.load(FH)
            for key in tmp.keys():
                if key in PermApiDictFromJson:
                    old_dict = PermApiDictFromJson[key]
                    for one_api in tmp[key]:
                        if one_api not in old_dict:
                            old_dict.append(one_api)
                    PermApiDictFromJson[key] = old_dict
                else:
                    PermApiDictFromJson[key] = tmp[key]
    PermApiDictFromJsonConvert={}
    for Perms in PermApiDictFromJson:
        for Api in range(len(PermApiDictFromJson[Perms])):
            ApiName=PermApiDictFromJson[Perms][Api][0].lower()+PermApiDictFromJson[Perms][Api][1].lower()
            PermApiDictFromJsonConvert[ApiName]=Perms


    return PermApiDictFromJsonConvert
graph_lock = threading.Lock()
def save_graph(graph, filename):
    """
    使用pickle保存图对象到文件
    """
    with graph_lock:
        with open(filename, 'wb') as f:
            pickle.dump(graph, f)
        print(f"图已成功保存到 {filename}")




def load_sensitive_graph(graph_path):
    try:
        if not os.path.exists(graph_path):
            return None

        with open(graph_path, 'rb') as f:
            tmp=pickle.load(f)
            graph = tmp[0] if isinstance(tmp,tuple) else tmp
        return graph
    except Exception as e:
        print(graph_path,f"加载图时出错: {e}")
        return None

def load_graph(filename):
    """
    从pickle文件加载图对象
    """
    try:
        with open(filename, 'rb') as f:
            graph = pickle.load(f)
        #print(f"图已成功从 {filename} 加载")
        return graph
    except Exception as e:
        print(filename,f"加载图时出错: {e}")
        return None

# 1. 查找最近的上游节点
def trace_trigger_components(graph, start_node, condition_func):
    """找到距离最近且满足条件的上游节点，并返回调用链"""
    visited = set()
    # 队列中存储节点和从起始点到该节点的路径
    queue = deque([(start_node, [start_node])])

    while queue:
        current, path = queue.popleft()

        for parent in graph.predecessors(current):
            if parent not in visited:
                visited.add(parent)
                # 创建从起始点到当前父节点的新路径
                new_path = [parent] + path

                if condition_func(parent):
                    # 返回满足条件的节点和完整的调用链
                    return parent, new_path

                queue.append((parent, new_path))

    return None, []  # 没有找到满足条件的节点
def activate_condition_func(node):
    if len(node.component):
        return True
    return False



from collections import deque

def find_downstream_nodes(graph, start_node, condition_func):
    """找到距离最近且满足条件的下游节点集合"""
    visited = set()
    result = set()
    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        for child in graph.successors(current):
            if child not in visited:
                visited.add(child)

                if condition_func(child):
                    result.add(child)
                else:
                    queue.append(child)

    return result

def get_minimal_connected_subgraph(graph, risk_nodes):
    """
    获取包含所有风险节点的最小连通子图
    """
    if not risk_nodes:
        return nx.DiGraph()

    # 包含所有风险节点
    all_nodes = set(risk_nodes)
    # 找到连接风险节点的最短路径
    for i, node1 in enumerate(risk_nodes):
        for node2 in risk_nodes[i+1:]:
            try:
                # 双向查找路径
                if nx.has_path(graph, node1, node2):
                    path = nx.shortest_path(graph, node1, node2)
                    all_nodes.update(path)
                if nx.has_path(graph, node2, node1):
                    path = nx.shortest_path(graph, node2, node1)
                    all_nodes.update(path)
            except:
                continue

    # 创建子图
    subgraph = graph.subgraph(all_nodes).copy()

    return subgraph

def get_trigger_components_and_paths(graph, risk_nodes):
    """
    获取触发组件AC和触发路径AI
    """
    AC = set()  # 触发组件集合
    AI = set()  # 触发路径节点集合

    for risk_node in risk_nodes:
        # 查找最近的触发组件
        trigger_component, trigger_path = trace_trigger_components(graph, risk_node, activate_condition_func)

        if trigger_component:
            AC.add(trigger_component)
            # 将触发路径中的所有节点添加到AI（排除风险节点本身）
            for node in trigger_path:
                if node != risk_node and node != trigger_component:
                    AI.add(node)

    return AC, AI

def get_background_nodes(graph, risk_nodes):
    """
    获取背景节点CM（存在特征的下游节点）
    """
    CM = set()

    for risk_node in risk_nodes:
        # 获取风险节点的所有下游节点
        downstream_nodes = get_downstream_tree(graph, risk_node)

        for node in downstream_nodes:
            # 检查节点是否具有特征（权限、敏感API、可疑API等）
            if has_features(node):
                CM.add(node)

    return CM
def has_features(node):
    return (len(node.permission) > 0 or
            len(node.sensitiveApi) > 0 or
            len(node.suspiciousApi) > 0 or
            len(node.url) > 0 or
            len(node.filterToken) > 0 or
            len(node.provider) > 0 or
            len(node.hardware_component) > 0)

def get_risk_components(graph, risk_nodes):
    # 1. 获取风险节点RM构成的最小连通图
    RM = get_minimal_connected_subgraph(graph, risk_nodes)
    # 2. 获取触发组件AC和触发路径AI
    AC, AI = get_trigger_components_and_paths(graph, risk_nodes)
    # 3. 获取背景节点CM
    CM = get_background_nodes(graph, risk_nodes)
    return RM, AC, AI, CM

def activate_condition_func(node):
    return bool(node.component)
def get_downstream_tree(G, start_node):
    return list(nx.descendants(G, start_node))
def get_nodes_by_class(graph,target_node):
    if target_node is None:
        return []
    nodes=[]

    class_name=target_node.name.split(';')[0]
    for node in graph.nodes:
        if class_name in node.name:
            nodes.append(node)
    return  nodes

def revert_smali_single(nodes):
    smali_list = []
    for node in nodes:
        class_name = node.name.split('->')[0]
        smali_code = f'.class public {class_name}\r\n.super {node.super}\r\n{node.body}'
        smali_list.append(smali_code)
    return smali_list
def revert_smali_batch(nodes):
    # 按类名分组
    class_groups = {}
    for node in nodes:
        class_name = node.name.split('->')[0]

        if class_name not in class_groups:
            class_groups[class_name] = {
                'super': node.super,
                'bodies': []
            }

        class_groups[class_name]['bodies'].append(node.body)

    # 为每个类生成smali代码
    smali_list = []

    for class_name, class_info in class_groups.items():
        # 合并相同类的body内容
        combined_body = '\r\n'.join(class_info['bodies'])

        smali_code = f'.class public {class_name}\r\n.super {class_info["super"]}\r\n{combined_body}'
        smali_list.append(smali_code)



    return smali_list
def list_2_str(list_i):
    if isinstance(list_i,list):
        content=''
        for one in list_i:
            content=(content+one+'\r\n')
        return content
    else:
        return list_i