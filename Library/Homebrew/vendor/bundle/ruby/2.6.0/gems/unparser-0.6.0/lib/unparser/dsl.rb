# frozen_string_literal: true

module Unparser
  # DSL to help defining emitters
  module DSL

  private

    def define_remaining_children(names)
      range = names.length..-1
      define_method(:remaining_children) do
        children[range]
      end
      private :remaining_children
    end

    def define_child(name, index)
      define_method(name) do
        children.at(index)
      end
      private name
    end

    def define_group(name, range)
      define_method(name) do
        children[range]
      end
      private(name)
      memoize(name)
    end

    def children(*names)
      define_remaining_children(names)

      names.each_with_index do |name, index|
        define_child(name, index)
      end
    end

  end # DSL
end # Unparser
