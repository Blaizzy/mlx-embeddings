# typed: strict
# frozen_string_literal: true

# Parlour type signature generator plugin for Homebrew DSL attributes.
class Attr < Parlour::Plugin
  sig { override.params(root: Parlour::RbiGenerator::Namespace).void }
  def generate(root)
    tree = T.let([], T::Array[T.untyped])
    Homebrew::Parlour.ast_list.each do |node|
      tree += find_custom_attr(node)
    end
    process_custom_attr(tree, root)
  end

  sig { override.returns(T.nilable(String)) }
  def strictness
    "strict"
  end

  private

  sig { params(node: Parser::AST::Node, list: T::Array[String]).returns(T::Array[String]) }
  def traverse_module_name(node, list = [])
    parent, name = node.children
    list = traverse_module_name(parent, list) if parent
    list << name.to_s
    list
  end

  sig { params(node: T.nilable(Parser::AST::Node)).returns(T.nilable(String)) }
  def extract_module_name(node)
    return if node.nil?

    traverse_module_name(node).join("::")
  end

  sig { params(node: Parser::AST::Node).returns(T::Array[T.untyped]) }
  def find_custom_attr(node)
    tree = T.let([], T::Array[T.untyped])
    children = node.children.dup

    if node.type == :begin
      children.each do |child|
        subtree = find_custom_attr(child)
        tree += subtree unless subtree.empty?
      end
    elsif node.type == :sclass
      subtree = find_custom_attr(node.children[1])
      return tree if subtree.empty?

      tree << [:sclass, subtree]
    elsif node.type == :class || node.type == :module
      element = []
      case node.type
      when :class
        element << :class
        element << extract_module_name(children.shift)
        element << extract_module_name(children.shift)
      when :module
        element << :module
        element << extract_module_name(children.shift)
      end

      body = children.shift
      return tree if body.nil?

      subtree = find_custom_attr(body)
      return tree if subtree.empty?

      element << subtree
      tree << element
    elsif node.type == :send && children.shift.nil?
      method_name = children.shift
      if [:attr_rw, :attr_predicate].include?(method_name)
        children.each do |name_node|
          tree << [method_name, name_node.children.first.to_s]
        end
      end
    end

    tree
  end

  sig { params(tree: T::Array[T.untyped], namespace: Parlour::RbiGenerator::Namespace, sclass: T::Boolean).void }
  def process_custom_attr(tree, namespace, sclass: false)
    tree.each do |node|
      type = node.shift
      case type
      when :sclass
        process_custom_attr(node.shift, namespace, sclass: true)
      when :class
        class_namespace = namespace.create_class(node.shift, superclass: node.shift)
        process_custom_attr(node.shift, class_namespace)
      when :module
        module_namespace = namespace.create_module(node.shift)
        process_custom_attr(node.shift, module_namespace)
      when :attr_rw
        name = node.shift
        name = "self.#{name}" if sclass
        namespace.create_method(name,
                                parameters:  [
                                  Parlour::RbiGenerator::Parameter.new("arg", type:    "T.untyped",
                                                                              default: "T.unsafe(nil)"),
                                ],
                                return_type: "T.untyped")
      when :attr_predicate
        name = node.shift
        name = "self.#{name}" if sclass
        namespace.create_method(name, return_type: "T::Boolean")
      else
        raise "Malformed tree."
      end
    end
  end
end
