# frozen_string_literal: true

require "rubocop/cop/style/mutable_constant"

module RuboCop
  module Cop
    module Sorbet
      module MutableConstantSorbetAwareBehaviour
        def self.prepended(base)
          base.def_node_matcher(:t_let, <<~PATTERN)
            (send (const nil? :T) :let $_constant _type)
          PATTERN
        end

        def on_assignment(value)
          t_let(value) do |constant|
            value = constant
          end

          super(value)
        end
      end
    end
  end
end

RuboCop::Cop::Style::MutableConstant.prepend(
  RuboCop::Cop::Sorbet::MutableConstantSorbetAwareBehaviour
)
