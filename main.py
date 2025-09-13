import sys
import logging
from functools import reduce
from lark import Tree, Token, Lark, UnexpectedInput, UnexpectedToken, UnexpectedCharacters

class Table(dict):
    """ Environment (scope) table with optional outer chaining. """

    def __init__(self, basic=False, symbol_names=None, symbol_values=None, outer=None):
        super().__init__()
        if symbol_names is None: symbol_names = []
        if symbol_values is None: symbol_values = []
        self.update(zip(symbol_names, symbol_values))
        self.outer = outer

        if basic:
            self.update({
                'print-num':  self.print_num,
                'print-bool': self.print_bool,
                'plus':          self.plus,
                'minus':          self.minus,
                'multiply':          self.multiply,
                'divide':          self.divide,
                'modulus':        self.modulus,
                'greater':          self.greater,
                'smaller':          self.smaller,
                'equal':          self.equal,
                'and_op':        self.and_op,
                'or_op':         self.or_op,
                'not_op':        self.not_op
            })

    def print_num(self, x):
        print(x)

    def print_bool(self, x):
        print("#t" if x else "#f")

    def plus(self, *args):
        for arg in args:
            if not isinstance(arg, int):
                raise TypeError("plus expects integers")
        return sum(args)

    def minus(self, a, b, *rest):
        # grammar enforces at least two expressions for minus
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("minus expects integers")
        val = a - b
        if rest:  # if more expressions, subtract them too
            for r in rest:
                val -= r
        return val

    def multiply(self, *args):
        val = 1
        for arg in args:
            if not isinstance(arg, int):
                raise TypeError("multiply expects integers")
            val *= arg
        return val

    def divide(self, a, b):
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("divide expects integers")
        return a // b

    def modulus(self, a, b):
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("mod expects integers")
        return a % b

    def greater(self, a, b):
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("greater expects integers")
        return a > b

    def smaller(self, a, b):
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("smaller expects integers")
        return a < b

    def equal(self, *args):
        # grammar allows ( = exp exp+ ) => at least 2 expressions
        first = args[0]
        for arg in args:
            if type(arg) != type(first):
                return False
        return all(arg == first for arg in args)

    def and_op(self, *args):
        for a in args:
            if not isinstance(a, bool):
                raise TypeError("and expects booleans")
        return all(args)

    def or_op(self, *args):
        for a in args:
            if not isinstance(a, bool):
                raise TypeError("or expects booleans")
        return any(args)

    def not_op(self, a):
        if not isinstance(a, bool):
            raise TypeError("not expects a boolean")
        return not a

    def find(self, name):
        if name in self:
            return self
        elif self.outer:
            return self.outer.find(name)
        else:
            raise NameError(f"'{name}' is not found")


class Function:
    """Represents a user-defined function (fun_exp)."""
    def __init__(self, param_list, body_def_stmts, body_expr, defining_env):
        self.param_list = param_list
        self.body_def_stmts = body_def_stmts
        self.body_expr = body_expr
        self.defining_env = defining_env

    def __call__(self, *args):
        # Create a new local environment chained to defining_env
        local_env = Table(outer=self.defining_env)
        for p, a in zip(self.param_list, args):
            local_env[p] = a

        # Evaluate def_stmts in the function body
        for def_stmt in self.body_def_stmts:
            interpret_AST(def_stmt, local_env)

        # Evaluate the final expression
        return interpret_AST(self.body_expr, local_env)


def interpret_AST(node, env=None):
    if env is None:
        env = Table(basic=True)

    # -- 1) If node is a Token, handle it by token type --
    if isinstance(node, Token):
        if node.type == 'NUMBER':
            return int(node.value)
        elif node.type == 'BOOLEAN':
            return True if node.value == '#t' else False
        elif node.type == 'ID':
            return env.find(node.value)[node.value]
        else:
            raise TypeError(f"Unexpected token type: {node.type}, value: {node.value}")

    # -- 2) If node is a Tree, dispatch based on node.data --
    if isinstance(node, Tree):
        data = node.data
        children = node.children

        def check_type(val, expected_type, error_message):
            if type(val) != expected_type:
                raise TypeError(error_message)

        if data == 'start':
            result = None
            for stmt in children:
                result = interpret_AST(stmt, env)
            return result

        elif data == 'print_num':
            val = interpret_AST(children[0], env)
            check_type(val, int, "Type Error: Expect 'number' but got 'boolean'")
            print(val)

        elif data == 'print_bool':
            val = interpret_AST(children[0], env)
            check_type(val, bool, "Type Error: Expect 'boolean' but got 'number'")
            print("#t" if val else "#f")

        # Arithmetic Operations
        elif data == 'plus':
            exps = [interpret_AST(c, env) for c in children]
            for exp in exps:
                check_type(exp, int, "Type Error: Expect 'number' but got 'boolean'")
            return sum(exps)

        elif data == 'minus':
            exps = [interpret_AST(c, env) for c in children]
            for exp in exps:
                check_type(exp, int, "Type Error: Expect 'number' but got 'boolean'")
            return exps[0] - sum(exps[1:])

        elif data == 'multiply':
            exps = [interpret_AST(c, env) for c in children]
            for exp in exps:
                check_type(exp, int, "Type Error: Expect 'number' but got 'boolean'")
            result = 1
            for exp in exps:
                result *= exp
            return result

        elif data == 'divide':
            left_val = interpret_AST(children[0], env)
            right_val = interpret_AST(children[1], env)
            check_type(left_val, int, "Type Error: Expect 'number' but got 'boolean'")
            check_type(right_val, int, "Type Error: Expect 'number' but got 'boolean'")
            return left_val // right_val

        elif data == 'modulus':
            left_val = interpret_AST(children[0], env)
            right_val = interpret_AST(children[1], env)
            check_type(left_val, int, "Type Error: Expect 'number' but got 'boolean'")
            check_type(right_val, int, "Type Error: Expect 'number' but got 'boolean'")
            return left_val % right_val

        # Comparison Operations
        elif data == 'greater':
            left_val = interpret_AST(children[0], env)
            right_val = interpret_AST(children[1], env)
            check_type(left_val, int, "Type Error: Expect 'number' but got 'boolean'")
            check_type(right_val, int, "Type Error: Expect 'number' but got 'boolean'")
            return left_val > right_val

        elif data == 'smaller':
            left_val = interpret_AST(children[0], env)
            right_val = interpret_AST(children[1], env)
            check_type(left_val, int, "Type Error: Expect 'number' but got 'boolean'")
            check_type(right_val, int, "Type Error: Expect 'number' but got 'boolean'")
            return left_val < right_val

        elif data == 'equal':
            exps = [interpret_AST(c, env) for c in children]
            first = exps[0]
            for exp in exps:
                if type(exp) != type(first):
                    return False
            return all(exp == first for exp in exps)

        # Logical Operations
        elif data == 'and_op':
            exps = [interpret_AST(c, env) for c in children]
            for exp in exps:
                check_type(exp, bool, "Type Error: Expect 'boolean' but got 'number'")
            return all(exps)

        elif data == 'or_op':
            exps = [interpret_AST(c, env) for c in children]
            for exp in exps:
                check_type(exp, bool, "Type Error: Expect 'boolean' but got 'number'")
            return any(exps)

        elif data == 'not_op':
            val = interpret_AST(children[0], env)
            check_type(val, bool, "Type Error: Expect 'boolean' but got 'number'")
            return not val

        # If Expression
        elif data == 'if_exp':
            condition = interpret_AST(children[0], env)
            check_type(condition, bool, "Type Error: Expect 'boolean' for condition in 'if' expression")
            return interpret_AST(children[1], env) if condition else interpret_AST(children[2], env)

        # Function Definition and Calls
        elif data == 'fun_exp':
            param_list = interpret_AST(children[0], env)
            body_def_stmts, body_expr = interpret_AST(children[1], env)
            return Function(param_list, body_def_stmts, body_expr, env)

        elif data == 'fun_ids':
            return [child.value for child in children]

        elif data == 'fun_body':
            def_stmts = children[:-1]
            last_exp = children[-1]
            return (def_stmts, last_exp)

        elif data == 'fun_call':
            func_node = children[0]
            arg_nodes = children[1:]
            func_obj = interpret_AST(func_node, env)
            arg_vals = [interpret_AST(arg, env) for arg in arg_nodes]
            return func_obj(*arg_vals)

        elif data == 'def_stmt':
            var = children[0]
            val = interpret_AST(children[1], env)
            env[var] = val

        elif data == 'variable':
            return children[0].value

        else:
            raise RuntimeError(f"Unknown parse node type: {data}")

    else:
        raise TypeError(f"Unknown node type: {type(node)}")

class Interpreter:
    def __init__(self, debug=False):
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        with open('mini_lisp.lark') as gramfile:
            self.parser = Lark(gramfile, start='start', parser='lalr', lexer='contextual')

    def interpret(self, code):
        try:
            tree = self.parser.parse(code)
        except (UnexpectedInput, UnexpectedToken, UnexpectedCharacters, SyntaxError) as e:
            raise SyntaxError(f"MIni-lisp syntax error")
        else:
            return interpret_AST(tree)


if __name__ == "__main__":  
    test_file = sys.argv[1]
    with open(test_file, 'r') as file:
        sample_code = file.read()
    interpreter = Interpreter(debug=True)
    interpreter.interpret(sample_code)
