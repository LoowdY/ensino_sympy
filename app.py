import streamlit as st
import sympy as sp
from sympy import (Symbol, symbols, solve, expand, factor, simplify, 
                  Matrix, sqrt, factorial, binomial, Interval,
                  And, Or, Not, Implies, Eq, solve_linear_system,
                  divisors, isprime, primefactors, gcd, lcm, 
                  FiniteSet, Union, Intersection, Complement)

def main():
    st.title("Tutorial de Matemática Discreta com SymPy")
    st.markdown("**Monitor:** João Renan | **Professor:** Pedro Girotto")
    
    menu = st.sidebar.selectbox(
        "Escolha o tópico",
        ["Introdução ao SymPy",
         "Teoria dos Conjuntos",
         "Lógica Matemática",
         "Relações e Funções",
         "Teoria dos Números",
         "Combinatória e Sequências",
         "Exercícios Práticos"]
    )
    
    if menu == "Introdução ao SymPy":
        introducao_sympy()
    elif menu == "Teoria dos Conjuntos":
        teoria_conjuntos()
    elif menu == "Lógica Matemática":
        logica_matematica()
    elif menu == "Relações e Funções":
        relacoes_funcoes()
    elif menu == "Teoria dos Números":
        teoria_numeros()
    elif menu == "Combinatória e Sequências":
        combinatoria_sequencias()
    else:
        exercicios_praticos()

def introducao_sympy():
    st.header("Introdução ao SymPy")
    
    st.subheader("1. Símbolos e Expressões Básicas")
    
    st.write("Primeiro, vamos ver como criar símbolos e expressões básicas:")
    st.code("""
    # Importando SymPy
    from sympy import symbols, expand, factor, simplify
    
    # Criando símbolos individuais
    x = Symbol('x')
    y = Symbol('y')
    
    # Criando múltiplos símbolos de uma vez
    a, b, c = symbols('a b c')
    
    # Criando expressões
    expr1 = x**2 + 2*x + 1
    expr2 = (x + y)**2
    
    # Manipulações básicas
    expandida = expand(expr2)      # x**2 + 2*x*y + y**2
    fatorada = factor(expr1)       # (x + 1)**2
    simplificada = simplify(expr)  # Simplifica expressões complexas
    """)
    
    # Área interativa
    st.write("Teste aqui! Digite uma expressão para manipular:")
    expr_str = st.text_input("Expressão (ex: x**2 + 2*x + 1):", "x**2 + 2*x + 1")
    
    try:
        x = Symbol('x')
        expr = eval(expr_str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original:")
            st.latex(sp.latex(expr))
            st.write("Expandida:")
            st.latex(sp.latex(expand(expr)))
        
        with col2:
            st.write("Fatorada:")
            st.latex(sp.latex(factor(expr)))
            st.write("Simplificada:")
            st.latex(sp.latex(simplify(expr)))
            
    except:
        st.error("Expressão inválida!")
    
    st.subheader("2. Equações e Sistemas")
    st.code("""
    # Resolvendo equações
    from sympy import solve
    
    # Equação única
    eq1 = x**2 - 4
    solucao = solve(eq1, x)  # [-2, 2]
    
    # Sistema de equações
    eq1 = Eq(2*x + y - 5, 0)
    eq2 = Eq(x - y - 1, 0)
    sistema = solve((eq1, eq2), (x, y))
    """)

def teoria_conjuntos():
    st.header("Teoria dos Conjuntos")
    
    st.subheader("1. Criando e Manipulando Conjuntos")
    st.code("""
    from sympy import FiniteSet, Interval, Union, Intersection
    
    # Conjuntos finitos
    A = FiniteSet(1, 2, 3, 4)
    B = FiniteSet(3, 4, 5, 6)
    
    # Operações com conjuntos
    uniao = Union(A, B)
    intersecao = Intersection(A, B)
    diferenca = Complement(A, B)
    
    # Intervalos
    I1 = Interval(0, 5)        # [0, 5]
    I2 = Interval.open(0, 5)   # (0, 5)
    I3 = Interval.Lopen(0, 5)  # (0, 5]
    
    # Verificação de pertinência
    pertence = 3 in A  # True
    """)
    
    # Área interativa
    st.write("Crie dois conjuntos e veja suas operações:")
    conj1_str = st.text_input("Conjunto A (números separados por espaço):", "1 2 3 4")
    conj2_str = st.text_input("Conjunto B (números separados por espaço):", "3 4 5 6")
    
    try:
        A = FiniteSet(*[int(x) for x in conj1_str.split()])
        B = FiniteSet(*[int(x) for x in conj2_str.split()])
        
        st.write("A ∪ B (União):")
        st.latex(sp.latex(Union(A, B)))
        
        st.write("A ∩ B (Interseção):")
        st.latex(sp.latex(Intersection(A, B)))
        
        st.write("A - B (Diferença):")
        st.latex(sp.latex(Complement(A, B)))
        
    except:
        st.error("Entrada inválida! Use números inteiros separados por espaço.")

def logica_matematica():
    st.header("Lógica Matemática")
    
    st.subheader("1. Proposições e Operações Lógicas")
    st.code("""
    from sympy import And, Or, Not, Implies, symbols, true, false
    
    # Criando variáveis proposicionais
    p, q, r = symbols('p q r')
    
    # Operações básicas
    conjuncao = And(p, q)         # p ∧ q
    disjuncao = Or(p, q)          # p ∨ q
    negacao = Not(p)              # ¬p
    implicacao = Implies(p, q)    # p → q
    
    # Expressões compostas
    expr1 = And(p, Or(q, Not(r))) # p ∧ (q ∨ ¬r)
    expr2 = Implies(And(p, q), r) # (p ∧ q) → r
    
    # Avaliando expressões
    resultado = expr1.subs({p: True, q: False, r: True})
    
    # Verificando tautologias
    expr3 = Or(p, Not(p))  # Sempre verdadeiro
    """)
    
    st.subheader("2. Construção de Tabelas Verdade")
    st.code("""
    # Função para gerar tabela verdade
    def tabela_verdade(expr, variaveis):
        n = len(variaveis)
        for valores in itertools.product([True, False], repeat=n):
            subs_dict = dict(zip(variaveis, valores))
            resultado = expr.subs(subs_dict)
            print(valores, "->", resultado)
    
    # Exemplo de uso
    expr = Implies(p, And(q, Not(r)))
    tabela_verdade(expr, [p, q, r])
    """)
    
    st.subheader("3. Teste Interativo")
    st.write("Construa uma expressão lógica:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        op1 = st.selectbox("Primeira operação:", ["p", "q", "Not p", "Not q"])
    
    with col2:
        conectivo = st.selectbox("Conectivo:", ["AND", "OR", "IMPLIES"])
    
    with col3:
        op2 = st.selectbox("Segunda operação:", ["p", "q", "Not p", "Not q"])
    
    try:
        p, q = symbols('p q')
        
        # Construir operandos
        def construir_op(op_str):
            if op_str == "p": return p
            if op_str == "q": return q
            if op_str == "Not p": return Not(p)
            if op_str == "Not q": return Not(q)
        
        op1_expr = construir_op(op1)
        op2_expr = construir_op(op2)
        
        # Construir expressão completa
        expr = {
            "AND": And(op1_expr, op2_expr),
            "OR": Or(op1_expr, op2_expr),
            "IMPLIES": Implies(op1_expr, op2_expr)
        }[conectivo]
        
        st.write("Expressão:")
        st.latex(sp.latex(expr))
        
        st.write("Tabela Verdade:")
        st.write("| p | q | Resultado |")
        st.write("|---|---|-----------|")
        for p_val in [True, False]:
            for q_val in [True, False]:
                result = expr.subs({p: p_val, q: q_val})
                st.write(f"| {p_val} | {q_val} | {result} |")
                
        # Verificar se é tautologia
        is_tautology = all(expr.subs({p: p_val, q: q_val}) 
                          for p_val in [True, False] 
                          for q_val in [True, False])
        
        if is_tautology:
            st.success("Esta expressão é uma tautologia!")
        else:
            st.info("Esta expressão não é uma tautologia.")
            
    except Exception as e:
        st.error(f"Erro ao construir expressão: {str(e)}")
    
    st.subheader("4. Equivalências Lógicas")
    st.code("""
    # Verificando equivalências lógicas
    expr1 = Implies(p, q)
    expr2 = Or(Not(p), q)
    
    # São equivalentes se expr1 <=> expr2 é tautologia
    equiv = And(Implies(expr1, expr2), Implies(expr2, expr1))
    
    # Ou usar simplify para verificar se a diferença é zero
    from sympy import simplify
    sao_equivalentes = simplify(expr1 - expr2) == 0
    """)
    
    st.subheader("5. Exemplos de Argumentos Lógicos")
    st.code("""
    # Modus Ponens
    premissa1 = Implies(p, q)  # p → q
    premissa2 = p              # p
    conclusao = q              # ∴ q
    
    # Verificar se o argumento é válido
    argumento = Implies(And(premissa1, premissa2), conclusao)
    
    # Se argumento.is_tautology é True, o argumento é válido
    
    # Modus Tollens
    premissa1 = Implies(p, q)  # p → q
    premissa2 = Not(q)         # ¬q
    conclusao = Not(p)         # ∴ ¬p
    """)
    
    st.write("""
    Exercícios sugeridos:
    1. Prove que p → q ≡ ¬p ∨ q
    2. Verifique se (p → q) ∧ (q → r) → (p → r) é uma tautologia
    3. Construa a tabela verdade para (p ∨ q) ∧ ¬(p ∧ q)
    4. Implemente o teste de validade para o Modus Tollens
    """)

def relacoes_funcoes():
    st.header("Relações e Funções")
    
    st.subheader("1. Relações")
    st.code("""
    from sympy import symbols, solve, Eq, Matrix
    
    # Definindo uma relação como matriz
    # 1 indica que os elementos estão relacionados, 0 caso contrário
    R = Matrix([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    
    # Verificar propriedades
    def eh_reflexiva(R):
        n = R.rows
        return all(R[i,i] == 1 for i in range(n))
    
    def eh_simetrica(R):
        n = R.rows
        return all(R[i,j] == R[j,i] for i in range(n) for j in range(n))
    
    def eh_transitiva(R):
        n = R.rows
        for i in range(n):
            for j in range(n):
                if R[i,j] == 1:
                    for k in range(n):
                        if R[j,k] == 1 and R[i,k] != 1:
                            return False
        return True
    """)
    
    st.write("Teste uma relação:")
    st.write("Digite a matriz de relação (1s e 0s separados por espaço, uma linha por vez):")
    
    try:
        linhas = []
        for i in range(3):
            linha = st.text_input(f"Linha {i+1}:", f"1 0 0")
            linhas.append([int(x) for x in linha.split()])
        
        R = Matrix(linhas)
        st.write("Matriz da relação:")
        st.latex(sp.latex(R))
        
        st.write("Propriedades:")
        st.write(f"Reflexiva: {all(R[i,i] == 1 for i in range(R.rows))}")
        st.write(f"Simétrica: {all(R[i,j] == R[j,i] for i in range(R.rows) for j in range(R.rows))}")
        
    except:
        st.error("Entrada inválida! Use números (0 ou 1) separados por espaço.")

    st.subheader("2. Funções")
    st.code("""
    from sympy import Function, solve, symbols
    
    x = symbols('x')
    
    # Definindo uma função
    f = 2*x + 1
    
    # Encontrar imagem
    x_val = 2
    imagem = f.subs(x, x_val)
    
    # Encontrar pré-imagem
    y_val = 5
    pre_imagem = solve(f - y_val, x)
    
    # Verificar se é injetora (todo y tem no máximo uma pré-imagem)
    def eh_injetora(f, x, dominio=Interval(-10, 10)):
        y = symbols('y')
        solucoes = solve(f - y, x)
        return len(solucoes) <= 1
    
    # Verificar se é sobrejetora (todo y tem pelo menos uma pré-imagem)
    def eh_sobrejetora(f, x, imagem=Interval(-10, 10)):
        y = symbols('y')
        solucoes = solve(f - y, x)
        return len(solucoes) >= 1
    """)
    
    st.write("Teste uma função:")
    func_str = st.text_input("Digite uma função em x:", "2*x + 1")
    
    try:
        x = symbols('x')
        f = eval(func_str)
        
        st.write("Sua função:")
        st.latex(sp.latex(f))
        
        # Teste com alguns valores
        st.write("Testando alguns valores:")
        for x_val in [-1, 0, 1]:
            y_val = f.subs(x, x_val)
            st.write(f"f({x_val}) = {y_val}")
            
        # Encontrar pré-imagem
        y_test = st.number_input("Encontrar pré-imagem para y =", value=5)
        pre_imagem = solve(f - y_test, x)
        st.write(f"Pré-imagem(ns) de {y_test}:", pre_imagem)
        
    except:
        st.error("Função inválida!")
    
    st.subheader("3. Composição de Funções")
    st.code("""
    # Compor funções
    f = 2*x + 1
    g = x**2
    
    # g ∘ f
    gof = g.subs(x, f)
    
    # f ∘ g
    fog = f.subs(x, g)
    """)
    
    st.write("Exemplo de composição:")
    f_str = st.text_input("f(x) =", "2*x + 1")
    g_str = st.text_input("g(x) =", "x**2")
    
    try:
        x = symbols('x')
        f = eval(f_str)
        g = eval(g_str)
        
        # Composição g ∘ f
        gof = g.subs(x, f)
        st.write("g ∘ f:")
        st.latex(sp.latex(gof))
        
        # Composição f ∘ g
        fog = f.subs(x, g)
        st.write("f ∘ g:")
        st.latex(sp.latex(fog))
        
    except:
        st.error("Funções inválidas!")

    st.write("""
    Exercícios sugeridos:
    1. Implemente uma função que verifica se uma relação é de equivalência
    2. Crie uma função que encontra a função inversa (se existir)
    3. Verifique se uma função é bijetora
    4. Implemente a composição de três funções
    """)

def teoria_numeros():
    st.header("Teoria dos Números")
    
    st.subheader("1. Divisibilidade e Números Primos")
    st.code("""
    from sympy import divisors, isprime, primefactors, gcd, lcm
    
    # Divisores de um número
    n = 12
    divs = divisors(n)          # [1, 2, 3, 4, 6, 12]
    
    # Verificar se é primo
    eh_primo = isprime(n)       # False
    
    # Fatores primos
    fatores = primefactors(n)   # [2, 3]
    
    # MDC e MMC
    a, b = 12, 18
    mdc = gcd(a, b)            # 6
    mmc = lcm(a, b)            # 36
    
    # Equação Diofantina
    from sympy.solvers.diophantine import diophantine
    from sympy import Symbol
    x, y = symbols('x y')
    eq = 3*x + 4*y - 10  # Resolve 3x + 4y = 10
    sol = diophantine(eq)
    """)
    
    st.write("Teste algumas operações:")
    num = st.number_input("Digite um número:", min_value=1, value=12)
    
    st.write(f"Divisores de {num}:", divisors(num))
    st.write(f"É primo? {isprime(num)}")
    st.write(f"Fatores primos: {primefactors(num)}")
    
    st.write("\nMDC e MMC:")
    num2 = st.number_input("Digite outro número:", min_value=1, value=18)
    st.write(f"MDC({num}, {num2}) = {gcd(num, num2)}")
    st.write(f"MMC({num}, {num2}) = {lcm(num, num2)}")
    
    st.subheader("2. Congruências e Aritmética Modular")
    st.code("""
    # Congruência modular
    from sympy import Mod
    
    # Calcular a ≡ b (mod n)
    a = 17
    n = 5
    resto = Mod(a, n)  # 2
    
    # Resolver congruências lineares
    # ax ≡ b (mod n)
    from sympy.solvers.diophantine import diop_linear
    x = Symbol('x')
    a, b, n = 3, 4, 7
    sol = solve_congruence((a, b, n))
    """)
    
    st.write("Teste congruência modular:")
    a = st.number_input("a =", value=17)
    n = st.number_input("n =", min_value=1, value=5)
    st.write(f"{a} ≡ {Mod(a, n)} (mod {n})")
    
    st.subheader("3. Funções Multiplicativas")
    st.code("""
    from sympy import totient, divisor_sigma
    
    # Função totiente de Euler
    n = 12
    phi = totient(n)  # Quantidade de números coprimos com n
    
    # Função sigma (soma dos divisores)
    sigma = divisor_sigma(n)  # Soma de todos os divisores
    
    # Função tau (quantidade de divisores)
    tau = divisor_sigma(n, 0)  # Número de divisores
    """)
    
    st.write(f"Função totiente φ({num}) = {totient(num)}")
    st.write(f"Soma dos divisores σ({num}) = {divisor_sigma(num)}")
    st.write(f"Quantidade de divisores τ({num}) = {divisor_sigma(num, 0)}")

def combinatoria_sequencias():
    st.header("Combinatória e Sequências")
    
    st.subheader("1. Permutações e Combinações")
    st.code("""
    from sympy import factorial, binomial
    
    # Fatorial
    n = 5
    fat = factorial(n)  # 5! = 120
    
    # Combinação
    n, r = 5, 2
    C = binomial(n, r)  # C(5,2) = 10
    
    # Permutação
    P = factorial(n) // factorial(n-r)  # P(5,2) = 20
    
    # Permutação com repetição
    from sympy import multinomial
    # multinomial(n1, n2, ...) = n!/(n1!*n2!*...)
    """)
    
    st.write("Calcule permutações e combinações:")
    n = st.number_input("n =", min_value=0, value=5)
    r = st.number_input("r =", min_value=0, max_value=n, value=2)
    
    st.write(f"C({n},{r}) = {binomial(n,r)}")
    st.write(f"P({n},{r}) = {factorial(n) // factorial(n-r)}")
    st.write(f"{n}! = {factorial(n)}")
    
    st.subheader("2. Sequências e Séries")
    st.code("""
    from sympy import summation, symbols, expand
    
    # Soma de sequência
    n = Symbol('n')
    soma = summation(n, (n, 1, 5))  # 1 + 2 + 3 + 4 + 5
    
    # Sequência aritmética
    a1, d = 1, 2  # primeiro termo e razão
    an = a1 + (n-1)*d  # termo geral
    Sn = n*(a1 + an)/2  # soma dos n primeiros termos
    
    # Sequência geométrica
    a1, q = 1, 2  # primeiro termo e razão
    an = a1*q**(n-1)  # termo geral
    Sn = a1*(1-q**n)/(1-q)  # soma dos n primeiros termos
    """)
    
    st.write("Análise de sequências:")
    seq_type = st.selectbox("Tipo de sequência:", 
                           ["Aritmética", "Geométrica"])
    
    a1 = st.number_input("Primeiro termo (a₁):", value=1)
    razao = st.number_input("Razão:", value=2)
    
    n = Symbol('n')
    if seq_type == "Aritmética":
        an = a1 + (n-1)*razao
        st.latex(f"a_n = {sp.latex(an)}")
        st.latex(f"S_n = {sp.latex(n*(2*a1 + (n-1)*razao)/2)}")
    else:
        an = a1*razao**(n-1)
        st.latex(f"a_n = {sp.latex(an)}")
        st.latex(f"S_n = {sp.latex(a1*(1-razao**n)/(1-razao))}")
        
    st.subheader("3. Princípio da Inclusão-Exclusão")
    st.code("""
    # |A ∪ B| = |A| + |B| - |A ∩ B|
    # |A ∪ B ∪ C| = |A| + |B| + |C| - |A ∩ B| - |A ∩ C| - |B ∩ C| + |A ∩ B ∩ C|
    
    from sympy import FiniteSet
    
    A = FiniteSet(1, 2, 3)
    B = FiniteSet(2, 3, 4)
    
    uniao = len(A.union(B))
    intersecao = len(A.intersection(B))
    PIE = len(A) + len(B) - len(A.intersection(B))
    """)

def exercicios_praticos():
    st.header("Exercícios Práticos")
    
    exercicio = st.selectbox(
        "Escolha o exercício",
        ["Teoria dos Números",
         "Combinatória",
         "Sequências",
         "Congruências",
         "Projeto Final"]
    )
    
    if exercicio == "Teoria dos Números":
        st.subheader("Exercício: Análise de Números")
        st.write("""
        Implemente uma função que recebe um número e retorna:
        1. Todos os seus divisores
        2. Sua fatoração em primos
        3. Se é abundante, deficiente ou perfeito
        4. Seus números coprimos até ele mesmo
        """)
        
        opcoes = {
            "Opção 1": """def analisar_numero(n):
    # Encontrar divisores
    divs = [i for i in range(1, n + 1) if n % i == 0]
    
    # Fatoração em primos
    def fatores_primos(n):
        fatores = []
        d = 2
        while n > 1:
            while n % d == 0:
                fatores.append(d)
                n //= d
            d += 1
        return fatores
    
    # Classificação
    soma_divs = sum(divs[:-1])  # soma dos divisores próprios
    if soma_divs == n:
        classificacao = "perfeito"
    elif soma_divs > n:
        classificacao = "abundante"
    else:
        classificacao = "deficiente"
    
    # Números coprimos
    from math import gcd
    coprimos = [i for i in range(1, n + 1) if gcd(i, n) == 1]
    
    return {
        'divisores': divs,
        'fatores_primos': fatores_primos(n),
        'classificacao': classificacao,
        'coprimos': coprimos
    }""",

            "Opção 2": """def analisar_numero(n):
    from sympy import divisors, primefactors, gcd
    
    divs = list(divisors(n))
    fat_primos = list(primefactors(n))
    
    soma_divs = sum(divs[:-1])
    if soma_divs == n:
        classificacao = "perfeito"
    elif soma_divs > n:
        classificacao = "abundante"
    else:
        classificacao = "deficiente"
    
    coprimos = [i for i in range(1, n + 1) if gcd(i, n) == 1]
    
    return {
        'divisores': divs,
        'fatores_primos': fat_primos,
        'classificacao': classificacao,
        'coprimos': coprimos
    }""",

            "Opção 3": """def analisar_numero(n):
    # Implementação incorreta
    divs = [1, n]
    fatores = [n]
    classificacao = "perfeito"
    coprimos = [1]
    
    return {
        'divisores': divs,
        'fatores_primos': fatores,
        'classificacao': classificacao,
        'coprimos': coprimos
    }""",

            "Opção 4": """def analisar_numero(n):
    # Implementação com erro de lógica
    divs = [i for i in range(1, n + 1)]
    fat_primos = [2, 3, 5]
    classificacao = "abundante"
    coprimos = list(range(1, n + 1))
    
    return {
        'divisores': divs,
        'fatores_primos': fat_primos,
        'classificacao': classificacao,
        'coprimos': coprimos
    }"""
        }
        
        resposta = st.radio("Escolha a implementação correta:", list(opcoes.keys()))
        st.code(opcoes[resposta])
        
        if st.button("Verificar Resposta"):
            st.write("### Análise das Implementações:")
            
            st.write("#### Opção 1:")
            st.write("""
            ✅ Implementação correta usando Python puro
            - Calcula divisores corretamente
            - Implementa fatoração em primos própria
            - Classifica o número corretamente
            - Encontra coprimos usando GCD
            """)
            
            st.write("#### Opção 2:")
            st.write("""
            ✅ Implementação correta usando SymPy
            - Usa funções built-in do SymPy (mais eficiente)
            - Mesma lógica de classificação
            - Resultado equivalente à Opção 1
            """)
            
            st.write("#### Opção 3:")
            st.write("""
            ❌ Implementação incorreta
            - Apenas considera 1 e n como divisores
            - Não faz fatoração em primos
            - Classificação sempre retorna "perfeito"
            - Considera apenas 1 como coprimo
            """)
            
            st.write("#### Opção 4:")
            st.write("""
            ❌ Implementação com erros lógicos
            - Considera todos os números como divisores
            - Usa lista fixa de fatores primos
            - Classificação sempre retorna "abundante"
            - Considera todos os números como coprimos
            """)
            
            st.write("### Gabarito:")
            st.success("As opções 1 e 2 estão corretas, mas a Opção 2 é mais eficiente por usar as funções do SymPy")
            
            # Demonstração
            st.write("### Demonstração:")
            n = 12
            exec(opcoes["Opção 2"])
            resultado = analisar_numero(n)
            st.write(f"Análise do número {n}:")
            for k, v in resultado.items():
                st.write(f"{k}: {v}")

    elif exercicio == "Combinatória":
        st.subheader("Exercício: Cálculos Combinatórios")
        st.write("""
        Implemente funções para calcular:
        1. Permutações com repetição
        2. Combinações com repetição
        3. Arranjos com repetição
        """)
        
        opcoes = {
            "Opção 1": """def calculos_combinatorios(n, elementos):
    from sympy import factorial, binomial
    
    # Permutação com repetição
    perm_rep = factorial(n) // prod(factorial(e) for e in elementos)
    
    # Combinação com repetição
    comb_rep = binomial(n + len(elementos) - 1, len(elementos) - 1)
    
    # Arranjo com repetição
    arr_rep = len(elementos) ** n
    
    return {
        'permutacao_rep': perm_rep,
        'combinacao_rep': comb_rep,
        'arranjo_rep': arr_rep
    }""",

            "Opção 2": """def calculos_combinatorios(n, elementos):
    # Implementação incorreta
    return {
        'permutacao_rep': n ** len(elementos),
        'combinacao_rep': n * len(elementos),
        'arranjo_rep': n + len(elementos)
    }""",

            "Opção 3": """def calculos_combinatorios(n, elementos):
    from sympy import factorial
    
    # Permutação com repetição - correto
    perm_rep = factorial(n) // prod(factorial(e) for e in elementos)
    
    # Combinação com repetição - incorreto
    comb_rep = factorial(n) // factorial(len(elementos))
    
    # Arranjo com repetição - incorreto
    arr_rep = factorial(n)
    
    return {
        'permutacao_rep': perm_rep,
        'combinacao_rep': comb_rep,
        'arranjo_rep': arr_rep
    }""",

            "Opção 4": """def calculos_combinatorios(n, elementos):
    from math import comb
    
    # Todas as fórmulas incorretas
    return {
        'permutacao_rep': comb(n, len(elementos)),
        'combinacao_rep': comb(n, 2),
        'arranjo_rep': n * len(elementos)
    }"""
        }
        
        resposta = st.radio("Escolha a implementação correta:", list(opcoes.keys()))
        st.code(opcoes[resposta])
        
        if st.button("Verificar Resposta"):
            st.write("### Análise das Implementações:")
            
            st.write("#### Opção 1:")
            st.write("""
            ✅ Implementação correta
            - Usa as fórmulas corretas para todos os cálculos
            - Utiliza funções do SymPy adequadamente
            - Considera repetições corretamente
            """)
            
            st.write("#### Opção 2:")
            st.write("""
            ❌ Implementação totalmente incorreta
            - Todas as fórmulas estão erradas
            - Não considera as repetições adequadamente
            """)
            
            st.write("#### Opção 3:")
            st.write("""
            ⚠️ Implementação parcialmente correta
            - Permutação com repetição está correta
            - Combinação e arranjo com repetição estão errados
            """)
            
            st.write("#### Opção 4:")
            st.write("""
            ❌ Implementação incorreta
            - Usa combinações simples em vez de com repetição
            - Fórmulas não correspondem aos conceitos pedidos
            """)
            
            st.write("### Gabarito:")
            st.success("A Opção 1 é a única completamente correta")
            
            # Demonstração
            st.write("### Demonstração:")
            n = 4
            elementos = [2, 1, 1]  # ex: AABB
            exec(opcoes["Opção 1"])
            resultado = calculos_combinatorios(n, elementos)
            st.write(f"Para n={n} e elementos={elementos}:")
            for k, v in resultado.items():
                st.write(f"{k}: {v}")

    elif exercicio == "Sequências":
        st.subheader("Exercício: Sequências Recursivas")
        st.write("""
        Implemente uma função que:
        1. Gere os n primeiros termos de Fibonacci
        2. Gere os n primeiros termos de uma PA
        3. Gere os n primeiros termos de uma PG
        4. Calcule a soma dos n primeiros termos
        """)
        
        opcoes = {
            "Opção 1": """def sequencias(n, a1=1, r=2):
    from sympy import fibonacci
    
    # Fibonacci usando SymPy
    fib = [fibonacci(i) for i in range(1, n+1)]
    
    # PA com primeiro termo a1 e razão r
    pa = [a1 + i*r for i in range(n)]
    soma_pa = n*(2*a1 + (n-1)*r)/2  # Soma da PA
    
    # PG com primeiro termo a1 e razão r
    pg = [a1 * r**i for i in range(n)]
    soma_pg = a1*(1-r**n)/(1-r) if r != 1 else a1*n  # Soma da PG
    
    return {
        'fibonacci': fib,
        'pa': pa,
        'pg': pg,
        'soma_pa': soma_pa,
        'soma_pg': soma_pg
    }""",

            "Opção 2": """def sequencias(n, a1=1, r=2):
    # Fibonacci recursivo (ineficiente)
    def fib(n):
        if n <= 2:
            return 1
        return fib(n-1) + fib(n-2)
    
    fibonacci = [fib(i) for i in range(1, n+1)]
    
    # PA e PG corretas
    pa = [a1 + i*r for i in range(n)]
    pg = [a1 * r**i for i in range(n)]
    
    # Somas calculadas manualmente
    soma_pa = sum(pa)
    soma_pg = sum(pg)
    
    return {
        'fibonacci': fibonacci,
        'pa': pa,
        'pg': pg,
        'soma_pa': soma_pa,
        'soma_pg': soma_pg
    }""",

            "Opção 3": """def sequencias(n, a1=1, r=2):
    # Todas as implementações incorretas
    fib = [i for i in range(n)]
    pa = [a1 * i for i in range(n)]
    pg = [a1 + r * i for i in range(n)]
    
    return {
        'fibonacci': fib,
        'pa': pa,
        'pg': pg,
        'soma_pa': sum(pa),
        'soma_pg': sum(pg)
    }""",

            "Opção 4": """def sequencias(n, a1=1, r=2):
    from sympy import fibonacci
    
    # Fibonacci correto
    fib = [fibonacci(i) for i in range(1, n+1)]
    
    # PA e PG trocadas
    pa = [a1 * r**i for i in range(n)]  # Esta é PG
    pg = [a1 + i*r for i in range(n)]   # Esta é PA
    
    return {
        'fibonacci': fib,
        'pa': pa,
        'pg': pg,
        'soma_pa': sum(pa),
        'soma_pg': sum(pg)
    }"""
        }
        
        resposta = st.radio("Escolha a implementação correta:", list(opcoes.keys()))
        st.code(opcoes[resposta])
        
        if st.button("Verificar Resposta"):
            st.write("### Análise das Implementações:")
            
            st.write("#### Opção 1:")
            st.write("""
            ✅ Implementação correta e eficiente
            - Usa fibonacci do SymPy (eficiente)
            - Implementa PA e PG corretamente
            - Usa fórmulas fechadas para as somas
            - Trata caso especial da PG quando r=1
            """)
            
            st.write("#### Opção 2:")
            st.write("""
            ⚠️ Parcialmente correta mas ineficiente
            - Fibonacci recursivo é ineficiente
            - PA e PG estão corretas
            - Calcula somas por iteração em vez de fórmulas fechadas
            """)
            
            st.write("#### Opção 3:")
            st.write("""
            ❌ Implementação totalmente incorreta
            - Fibonacci é apenas sequência natural
            - PA usa multiplicação em vez de adição
            - PG usa adição em vez de multiplicação
            """)
            
            st.write("#### Opção 4:")
            st.write("""
            ❌ Implementação com erros conceituais
            - Fibonacci está correto
            - PA e PG estão trocadas
            - Não usa fórmulas para somas
            """)
            
            st.write("### Gabarito:")
            st.success("A Opção 1 é a correta e mais eficiente")
            
            # Demonstração
            st.write("### Demonstração:")
            n = 5
            a1 = 2
            r = 3
            exec(opcoes["Opção 1"])
            resultado = sequencias(n, a1, r)
            st.write(f"Para n={n}, a1={a1}, r={r}:")
            for k, v in resultado.items():
                st.write(f"{k}: {v}")

    elif exercicio == "Congruências":
        st.subheader("Exercício: Sistema de Congruências")
        st.write("""
        Implemente uma função que resolva um sistema de congruências:
        x ≡ a₁ (mod m₁)
        x ≡ a₂ (mod m₂)
        x ≡ a₃ (mod m₃)
        """)
        
        opcoes = {
            "Opção 1": """def resolver_congruencias(a_list, m_list):
    from sympy.ntheory.modular import solve_congruence
    from functools import reduce
    from operator import mul
    
    # Resolve usando o Teorema Chinês do Resto
    solucao = solve_congruence(*zip(a_list, m_list))
    
    # Calcular módulo comum (M = m₁ * m₂ * m₃)
    M = reduce(mul, m_list)
    
    return {
        'solucao': solucao[0],
        'modulo': solucao[1] if len(solucao) > 1 else M,
        'solucao_geral': f"x ≡ {solucao[0]} (mod {M})"
    }""",

            "Opção 2": """def resolver_congruencias(a_list, m_list):
    # Implementação manual do Teorema Chinês do Resto
    from sympy import gcd, lcm
    
    # Verificar se os módulos são coprimos
    for i in range(len(m_list)):
        for j in range(i+1, len(m_list)):
            if gcd(m_list[i], m_list[j]) != 1:
                return "Módulos não são coprimos"
    
    M = 1
    for m in m_list:
        M *= m
    
    resultado = 0
    for i in range(len(m_list)):
        Mi = M // m_list[i]
        yi = pow(Mi, -1, m_list[i])
        resultado += a_list[i] * Mi * yi
    
    return {
        'solucao': resultado % M,
        'modulo': M,
        'solucao_geral': f"x ≡ {resultado % M} (mod {M})"
    }""",

            "Opção 3": """def resolver_congruencias(a_list, m_list):
    # Implementação incorreta
    solucao = sum(a_list) % sum(m_list)
    modulo = max(m_list)
    
    return {
        'solucao': solucao,
        'modulo': modulo,
        'solucao_geral': f"x ≡ {solucao} (mod {modulo})"
    }""",

            "Opção 4": """def resolver_congruencias(a_list, m_list):
    # Implementação que só funciona para dois casos
    if len(a_list) != 2 or len(m_list) != 2:
        return "Erro: só funciona para duas congruências"
    
    a1, a2 = a_list
    m1, m2 = m_list
    
    for x in range(m1 * m2):
        if x % m1 == a1 and x % m2 == a2:
            return {
                'solucao': x,
                'modulo': m1 * m2,
                'solucao_geral': f"x ≡ {x} (mod {m1 * m2})"
            }
            
    return "Sem solução" """
        }
        
        resposta = st.radio("Escolha a implementação correta:", list(opcoes.keys()))
        st.code(opcoes[resposta])
        
        if st.button("Verificar Resposta"):
            st.write("### Análise das Implementações:")
            
            st.write("#### Opção 1:")
            st.write("""
            ✅ Implementação ideal usando SymPy
            - Usa função solve_congruence do SymPy
            - Implementa o Teorema Chinês do Resto
            - Retorna solução completa e módulo
            - Mais eficiente e robusta
            """)
            
            st.write("#### Opção 2:")
            st.write("""
            ✅ Implementação manual correta
            - Implementa TCR manualmente
            - Verifica se módulos são coprimos
            - Mais complexa mas educativa
            - Pode ser menos eficiente
            """)
            
            st.write("#### Opção 3:")
            st.write("""
            ❌ Implementação totalmente incorreta
            - Não usa o Teorema Chinês do Resto
            - Operações sem sentido matemático
            - Não resolve o sistema de congruências
            """)
            
            st.write("#### Opção 4:")
            st.write("""
            ⚠️ Implementação limitada e ineficiente
            - Só funciona para duas congruências
            - Usa força bruta
            - Ineficiente para números grandes
            - Não usa TCR
            """)
            
            st.write("### Gabarito:")
            st.success("""
            Opções 1 e 2 estão corretas:
            - Opção 1 é melhor para uso prático
            - Opção 2 é boa para entender o algoritmo
            """)
            
            # Demonstração
            st.write("### Demonstração:")
            a_list = [2, 3, 2]
            m_list = [3, 5, 7]
            exec(opcoes["Opção 1"])
            resultado = resolver_congruencias(a_list, m_list)
            st.write(f"Para o sistema:")
            for i in range(len(a_list)):
                st.write(f"x ≡ {a_list[i]} (mod {m_list[i]})")
            st.write("Solução:")
            for k, v in resultado.items():
                st.write(f"{k}: {v}")

    else:
        st.subheader("Projeto Final: Calculadora de Matemática Discreta")
        st.write("""
        Implemente uma calculadora completa de matemática discreta que integre todos os conceitos anteriores.
        
        Sua calculadora deve incluir:
        1. Análise de números
        2. Cálculos combinatórios
        3. Sequências e séries
        4. Congruências modulares
        5. Interface amigável
        """)
        
        if st.button("Ver Exemplo de Implementação"):
            st.write("Aqui está um exemplo de como estruturar sua calculadora:")
            st.code("""
def calculadora_mat_discreta():
    st.sidebar.title("Calculadora Matemática Discreta")
    opcao = st.sidebar.selectbox("Escolha a operação",
        ["Análise de Números",
         "Combinatória",
         "Sequências",
         "Congruências"])
    
    if opcao == "Análise de Números":
        n = st.number_input("Digite um número:", min_value=1)
        resultado = analisar_numero(n)
        st.write(resultado)
    
    elif opcao == "Combinatória":
        n = st.number_input("n:", min_value=1)
        elementos = st.text_input("Elementos (separados por vírgula):")
        elementos = [int(x) for x in elementos.split(",")]
        resultado = calculos_combinatorios(n, elementos)
        st.write(resultado)
    
    elif opcao == "Sequências":
        n = st.number_input("Quantidade de termos:", min_value=1)
        a1 = st.number_input("Primeiro termo:")
        r = st.number_input("Razão:")
        resultado = sequencias(n, a1, r)
        st.write(resultado)
    
    else:  # Congruências
        a_list = eval(st.text_input("Lista de restos [a₁, a₂, ...]:", "[2,3,2]"))
        m_list = eval(st.text_input("Lista de módulos [m₁, m₂, ...]:", "[3,5,7]"))
        resultado = resolver_congruencias(a_list, m_list)
        st.write(resultado)
""")

if __name__ == "__main__":
    main()
