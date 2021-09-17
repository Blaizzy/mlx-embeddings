# typed: strict
# frozen_string_literal: true

require "unparser"

module RBI
  class ParseError < StandardError
    extend T::Sig

    sig { returns(Loc) }
    attr_reader :location

    sig { params(message: String, location: Loc).void }
    def initialize(message, location)
      super(message)
      @location = location
    end
  end

  class Parser
    extend T::Sig

    # opt-in to most recent AST format
    ::Parser::Builders::Default.emit_lambda               = true
    ::Parser::Builders::Default.emit_procarg0             = true
    ::Parser::Builders::Default.emit_encoding             = true
    ::Parser::Builders::Default.emit_index                = true
    ::Parser::Builders::Default.emit_arg_inside_procarg0  = true

    sig { params(string: String).returns(Tree) }
    def self.parse_string(string)
      Parser.new.parse_string(string)
    end

    sig { params(path: String).returns(Tree) }
    def self.parse_file(path)
      Parser.new.parse_file(path)
    end

    sig { params(string: String).returns(Tree) }
    def parse_string(string)
      parse(string, file: "-")
    end

    sig { params(path: String).returns(Tree) }
    def parse_file(path)
      parse(::File.read(path), file: path)
    end

    private

    sig { params(content: String, file: String).returns(Tree) }
    def parse(content, file:)
      node, comments = Unparser.parse_with_comments(content)
      assoc = ::Parser::Source::Comment.associate_locations(node, comments)
      builder = TreeBuilder.new(file: file, comments: assoc)
      builder.separate_header_comments
      builder.visit(node)
      builder.assoc_dangling_comments(comments)
      builder.tree
    rescue ::Parser::SyntaxError => e
      raise ParseError.new(e.message, Loc.from_ast_loc(file, e.diagnostic.location))
    end
  end

  class ASTVisitor
    extend T::Helpers
    extend T::Sig

    abstract!

    sig { params(nodes: T::Array[AST::Node]).void }
    def visit_all(nodes)
      nodes.each { |node| visit(node) }
    end

    sig { abstract.params(node: T.nilable(AST::Node)).void }
    def visit(node); end

    private

    sig { params(node: AST::Node).returns(String) }
    def parse_name(node)
      T.must(ConstBuilder.visit(node))
    end

    sig { params(node: AST::Node).returns(String) }
    def parse_expr(node)
      Unparser.unparse(node)
    end
  end

  class TreeBuilder < ASTVisitor
    extend T::Sig

    sig { returns(Tree) }
    attr_reader :tree

    sig do
      params(
        file: String,
        comments: T::Hash[::Parser::Source::Map, T::Array[::Parser::Source::Comment]]
      ).void
    end
    def initialize(file:, comments: {})
      super()
      @file = file
      @comments = comments
      @tree = T.let(Tree.new, Tree)
      @scopes_stack = T.let([@tree], T::Array[Tree])
      @last_sigs = T.let([], T::Array[RBI::Sig])
    end

    sig { override.params(node: T.nilable(Object)).void }
    def visit(node)
      return unless node.is_a?(AST::Node)
      case node.type
      when :module, :class, :sclass
        scope = parse_scope(node)
        current_scope << scope
        @scopes_stack << scope
        visit_all(node.children)
        @scopes_stack.pop
      when :casgn
        current_scope << parse_const_assign(node)
      when :def, :defs
        current_scope << parse_def(node)
      when :send
        node = parse_send(node)
        current_scope << node if node
      when :block
        node = parse_block(node)
        if node.is_a?(Sig)
          @last_sigs << node
        elsif node
          current_scope << node
        end
      else
        visit_all(node.children)
      end
    end

    sig { void }
    def separate_header_comments
      return if @comments.empty?

      keep = []
      node = T.must(@comments.keys.first)
      comments = T.must(@comments.values.first)

      last_line = T.let(nil, T.nilable(Integer))
      comments.reverse.each do |comment|
        comment_line = comment.location.last_line

        break if last_line && comment_line < last_line - 1 ||
          !last_line && comment_line < node.first_line - 1

        keep << comment
        last_line = comment_line
      end

      @comments[node] = keep.reverse
    end

    sig { params(comments: T::Array[::Parser::Source::Comment]).void }
    def assoc_dangling_comments(comments)
      last_line = T.let(nil, T.nilable(Integer))
      (comments - @comments.values.flatten).each do |comment|
        comment_line = comment.location.last_line
        text = comment.text[1..-1].strip
        loc = Loc.from_ast_loc(@file, comment.location)

        if last_line && comment_line > last_line + 1
          # Preserve empty lines in file headers
          tree.comments << EmptyComment.new(loc: loc)
        end

        tree.comments << Comment.new(text, loc: loc)
        last_line = comment_line
      end
    end

    private

    sig { params(node: AST::Node).returns(Scope) }
    def parse_scope(node)
      loc = node_loc(node)
      comments = node_comments(node)

      case node.type
      when :module
        name = parse_name(node.children[0])
        Module.new(name, loc: loc, comments: comments)
      when :class
        name = parse_name(node.children[0])
        superclass_name = ConstBuilder.visit(node.children[1])
        Class.new(name, superclass_name: superclass_name, loc: loc, comments: comments)
      when :sclass
        SingletonClass.new(loc: loc, comments: comments)
      else
        raise ParseError.new("Unsupported scope node type `#{node.type}`", loc)
      end
    end

    sig { params(node: AST::Node).returns(RBI::Node) }
    def parse_const_assign(node)
      node_value = node.children[2]
      if struct_definition?(node_value)
        parse_struct(node)
      else
        name = parse_name(node)
        value = parse_expr(node_value)
        loc = node_loc(node)
        comments = node_comments(node)
        Const.new(name, value, loc: loc, comments: comments)
      end
    end

    sig { params(node: AST::Node).returns(Method) }
    def parse_def(node)
      loc = node_loc(node)

      case node.type
      when :def
        Method.new(
          node.children[0].to_s,
          params: node.children[1].children.map { |child| parse_param(child) },
          sigs: current_sigs,
          loc: loc,
          comments: node_comments(node)
        )
      when :defs
        Method.new(
          node.children[1].to_s,
          params: node.children[2].children.map { |child| parse_param(child) },
          is_singleton: true,
          sigs: current_sigs,
          loc: loc,
          comments: node_comments(node)
        )
      else
        raise ParseError.new("Unsupported def node type `#{node.type}`", loc)
      end
    end

    sig { params(node: AST::Node).returns(Param) }
    def parse_param(node)
      name = node.children[0].to_s
      loc = node_loc(node)
      comments = node_comments(node)

      case node.type
      when :arg
        ReqParam.new(name, loc: loc, comments: comments)
      when :optarg
        value = parse_expr(node.children[1])
        OptParam.new(name, value, loc: loc, comments: comments)
      when :restarg
        RestParam.new(name, loc: loc, comments: comments)
      when :kwarg
        KwParam.new(name, loc: loc, comments: comments)
      when :kwoptarg
        value = parse_expr(node.children[1])
        KwOptParam.new(name, value, loc: loc, comments: comments)
      when :kwrestarg
        KwRestParam.new(name, loc: loc, comments: comments)
      when :blockarg
        BlockParam.new(name, loc: loc, comments: comments)
      else
        raise ParseError.new("Unsupported param node type `#{node.type}`", loc)
      end
    end

    sig { params(node: AST::Node).returns(T.nilable(RBI::Node)) }
    def parse_send(node)
      recv = node.children[0]
      return nil if recv && recv != :self

      method_name = node.children[1]
      loc = node_loc(node)
      comments = node_comments(node)

      case method_name
      when :attr_reader
        symbols = node.children[2..-1].map { |child| child.children[0] }
        AttrReader.new(*symbols, sigs: current_sigs, loc: loc, comments: comments)
      when :attr_writer
        symbols = node.children[2..-1].map { |child| child.children[0] }
        AttrWriter.new(*symbols, sigs: current_sigs, loc: loc, comments: comments)
      when :attr_accessor
        symbols = node.children[2..-1].map { |child| child.children[0] }
        AttrAccessor.new(*symbols, sigs: current_sigs, loc: loc, comments: comments)
      when :include
        names = node.children[2..-1].map { |child| parse_name(child) }
        Include.new(*names, loc: loc, comments: comments)
      when :extend
        names = node.children[2..-1].map { |child| parse_name(child) }
        Extend.new(*names, loc: loc, comments: comments)
      when :abstract!, :sealed!, :interface!
        Helper.new(method_name.to_s.delete_suffix("!"), loc: loc, comments: comments)
      when :mixes_in_class_methods
        names = node.children[2..-1].map { |child| parse_name(child) }
        MixesInClassMethods.new(*names, loc: loc, comments: comments)
      when :public, :protected, :private
        visibility = Visibility.new(method_name, loc: loc)
        nested_node = node.children[2]
        case nested_node&.type
        when :def, :defs
          method = parse_def(nested_node)
          method.visibility = visibility
          method
        when :send
          snode = parse_send(nested_node)
          raise ParseError.new("Unexpected token `private` before `#{nested_node.type}`", loc) unless snode.is_a?(Attr)
          snode.visibility = visibility
          snode
        when nil
          visibility
        else
          raise ParseError.new("Unexpected token `private` before `#{nested_node.type}`", loc)
        end
      when :prop
        name, type, default_value = parse_tstruct_prop(node)
        TStructProp.new(name, type, default: default_value, loc: loc, comments: comments)
      when :const
        name, type, default_value = parse_tstruct_prop(node)
        TStructConst.new(name, type, default: default_value, loc: loc, comments: comments)
      else
        raise ParseError.new("Unsupported send node with name `#{method_name}`", loc)
      end
    end

    sig { params(node: AST::Node).returns(T.nilable(RBI::Node)) }
    def parse_block(node)
      name = node.children[0].children[1]

      case name
      when :sig
        parse_sig(node)
      when :enums
        parse_enum(node)
      else
        raise ParseError.new("Unsupported block node type `#{name}`", node_loc(node))
      end
    end

    sig { params(node: AST::Node).returns(T::Boolean) }
    def struct_definition?(node)
      (node.type == :send && node.children[0]&.type == :const && node.children[0].children[1] == :Struct) ||
        (node.type == :block && struct_definition?(node.children[0]))
    end

    sig { params(node: AST::Node).returns(RBI::Struct) }
    def parse_struct(node)
      name = parse_name(node)
      loc = node_loc(node)
      comments = node_comments(node)

      send = node.children[2]
      body = []

      if send.type == :block
        if send.children[2].type == :begin
          body = send.children[2].children
        else
          body << send.children[2]
        end
        send = send.children[0]
      end

      members = []
      keyword_init = T.let(false, T::Boolean)
      send.children[2..].each do |child|
        if child.type == :sym
          members << child.children[0]
        elsif child.type == :kwargs
          pair = child.children[0]
          if pair.children[0].children[0] == :keyword_init
            keyword_init = true if pair.children[1].type == :true
          end
        end
      end

      struct = Struct.new(name, members: members, keyword_init: keyword_init, loc: loc, comments: comments)
      @scopes_stack << struct
      visit_all(body)
      @scopes_stack.pop

      struct
    end

    sig { params(node: AST::Node).returns([String, String, T.nilable(String)]) }
    def parse_tstruct_prop(node)
      name = node.children[2].children[0].to_s
      type = parse_expr(node.children[3])
      has_default = node.children[4]
        &.children&.fetch(0, nil)
        &.children&.fetch(0, nil)
        &.children&.fetch(0, nil) == :default
      default_value = if has_default
        parse_expr(node.children.fetch(4, nil)
          &.children&.fetch(0, nil)
          &.children&.fetch(1, nil))
      end
      [name, type, default_value]
    end

    sig { params(node: AST::Node).returns(Sig) }
    def parse_sig(node)
      sig = SigBuilder.build(node)
      sig.loc = node_loc(node)
      sig
    end

    sig { params(node: AST::Node).returns(TEnumBlock) }
    def parse_enum(node)
      enum = TEnumBlock.new
      node.children[2].children.each do |child|
        enum << parse_name(child)
      end
      enum.loc = node_loc(node)
      enum
    end

    sig { params(node: AST::Node).returns(Loc) }
    def node_loc(node)
      Loc.from_ast_loc(@file, node.location)
    end

    sig { params(node: AST::Node).returns(T::Array[Comment]) }
    def node_comments(node)
      comments = @comments[node.location]
      return [] unless comments
      comments.map do |comment|
        text = comment.text[1..-1].strip
        loc = Loc.from_ast_loc(@file, comment.location)
        Comment.new(text, loc: loc)
      end
    end

    sig { returns(Tree) }
    def current_scope
      T.must(@scopes_stack.last) # Should never be nil since we create a Tree as the root
    end

    sig { returns(T::Array[Sig]) }
    def current_sigs
      sigs = @last_sigs.dup
      @last_sigs.clear
      sigs
    end
  end

  class ConstBuilder < ASTVisitor
    extend T::Sig

    sig { params(node: T.nilable(AST::Node)).returns(T.nilable(String)) }
    def self.visit(node)
      v = ConstBuilder.new
      v.visit(node)
      return nil if v.names.empty?
      v.names.join("::")
    end

    sig { returns(T::Array[String]) }
    attr_accessor :names

    sig { void }
    def initialize
      super
      @names = T.let([], T::Array[String])
    end

    sig { override.params(node: T.nilable(AST::Node)).void }
    def visit(node)
      return unless node
      case node.type
      when :const, :casgn
        visit(node.children[0])
        @names << node.children[1].to_s
      when :cbase
        @names << ""
      when :sym
        @names << ":#{node.children[0]}"
      end
    end
  end

  class SigBuilder < ASTVisitor
    extend T::Sig

    sig { params(node: AST::Node).returns(Sig) }
    def self.build(node)
      v = SigBuilder.new
      v.visit_all(node.children[2..-1])
      v.current
    end

    sig { returns(Sig) }
    attr_accessor :current

    sig { void }
    def initialize
      super
      @current = T.let(Sig.new, Sig)
    end

    sig { override.params(node: T.nilable(AST::Node)).void }
    def visit(node)
      return unless node
      case node.type
      when :send
        visit_send(node)
      end
    end

    sig { params(node: AST::Node).void }
    def visit_send(node)
      visit(node.children[0]) if node.children[0]
      name = node.children[1]
      case name
      when :abstract
        @current.is_abstract = true
      when :override
        @current.is_override = true
      when :overridable
        @current.is_overridable = true
      when :checked
        @current.checked = node.children[2].children[0]
      when :type_parameters
        node.children[2..-1].each do |child|
          @current.type_params << child.children[0].to_s
        end
      when :params
        node.children[2].children.each do |child|
          name = child.children[0].children[0].to_s
          type = parse_expr(child.children[1])
          @current << SigParam.new(name, type)
        end
      when :returns
        @current.return_type = parse_expr(node.children[2])
      when :void
        @current.return_type = nil
      else
        raise "#{node.location.line}: Unhandled #{name}"
      end
    end
  end

  class Loc
    sig { params(file: String, ast_loc: T.any(::Parser::Source::Map, ::Parser::Source::Range)).returns(Loc) }
    def self.from_ast_loc(file, ast_loc)
      Loc.new(
        file: file,
        begin_line: ast_loc.line,
        begin_column: ast_loc.column,
        end_line: ast_loc.last_line,
        end_column: ast_loc.last_column
      )
    end
  end
end
