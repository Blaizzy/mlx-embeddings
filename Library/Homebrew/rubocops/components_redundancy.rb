# typed: true
# frozen_string_literal: true

require "rubocops/extend/formula"

module RuboCop
  module Cop
    module FormulaAudit
      # This cop checks if redundant components are present and for other component errors.
      #
      # - `url|checksum|mirror` should be inside `stable` block
      # - `head` and `head do` should not be simultaneously present
      # - `bottle :unneeded`/`:disable` and `bottle do` should not be simultaneously present
      # - `stable do` should not be present without a `head` spec
      #
      # @api private
      class ComponentsRedundancy < FormulaCop
        HEAD_MSG = "`head` and `head do` should not be simultaneously present"
        BOTTLE_MSG = "`bottle :modifier` and `bottle do` should not be simultaneously present"
        STABLE_MSG = "`stable do` should not be present without a `head` spec"

        def audit_formula(_node, _class_node, _parent_class_node, body_node)
          urls = find_method_calls_by_name(body_node, :url)

          urls.each do |url|
            url.arguments.each do |arg|
              next if arg.class != RuboCop::AST::HashNode

              url_args = arg.keys.each.map(&:value)
              if method_called?(body_node, :sha256) && url_args.include?(:tag) && url_args.include?(:revision)
                problem "Do not use both sha256 and tag/revision."
              end
            end
          end

          stable_block = find_block(body_node, :stable)
          if stable_block
            [:url, :sha256, :mirror].each do |method_name|
              problem "`#{method_name}` should be put inside `stable` block" if method_called?(body_node, method_name)
            end
          end

          problem HEAD_MSG if method_called?(body_node, :head) &&
                              find_block(body_node, :head)

          problem BOTTLE_MSG if method_called?(body_node, :bottle) &&
                                find_block(body_node, :bottle)

          return if method_called?(body_node, :head) ||
                    find_block(body_node, :head)

          problem STABLE_MSG if stable_block
        end
      end
    end
  end
end
