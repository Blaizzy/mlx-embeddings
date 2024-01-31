# typed: strict
# frozen_string_literal: true

require "method_source"
require "rubocop"
require_relative "../../../rubocops"

module Tapioca
  module Compilers
    class Rubocop < Tapioca::Dsl::Compiler
      # FIXME: Enable cop again when https://github.com/sorbet/sorbet/issues/3532 is fixed.
      # rubocop:disable Style/MutableConstant
      ConstantType = type_member { { fixed: T::Class[T.anything] } }
      # rubocop:enable Style/MutableConstant

      sig { override.returns(T::Enumerable[Module]) }
      def self.gather_constants
        [
          RuboCop::Cop::Cask::Variables,
          RuboCop::Cop::Homebrew::Blank,
          RuboCop::Cop::Homebrew::CompactBlank,
          RuboCop::Cop::Homebrew::MoveToExtendOS,
          RuboCop::Cop::Homebrew::NegateInclude,
          RuboCop::Cop::Homebrew::Presence,
          RuboCop::Cop::Homebrew::Present,
          RuboCop::Cop::Homebrew::SafeNavigationWithBlank,
          RuboCop::Cop::FormulaAudit::ComponentsOrder,
          RuboCop::Cop::FormulaAudit::DependencyOrder,
          RuboCop::Cop::FormulaAudit::DeprecateDisableDate,
          RuboCop::Cop::FormulaAudit::DeprecateDisableReason,
          RuboCop::Cop::FormulaAudit::Licenses,
          RuboCop::Cop::FormulaAudit::OptionDeclarations,
          RuboCop::Cop::FormulaAudit::GenerateCompletionsDSL,
          RuboCop::Cop::FormulaAudit::GitUrls,
          RuboCop::Cop::FormulaAudit::Miscellaneous,
          RuboCop::Cop::FormulaAudit::Patches,
          RuboCop::Cop::FormulaAudit::Test,
          RuboCop::Cop::FormulaAudit::Text,
          RuboCop::Cop::FormulaAuditStrict::GitUrls,
          RuboCop::Cop::FormulaAuditStrict::Text,
          RuboCop::Cop::FormulaCop,
          RuboCop::Cop::OnSystemConditionalsHelper,
        ]
      end

      sig { override.void }
      def decorate
        root.create_path(constant) do |klass|
          # For each encrypted attribute we find in the class
          constant.instance_methods(false).each do |method_name|
            source = constant.instance_method(method_name).source.lstrip
            # https://www.rubydoc.info/gems/rubocop-ast/RuboCop/AST/NodePattern/Macros
            # https://github.com/rubocop/rubocop-ast/blob/master/lib/rubocop/ast/node_pattern.rb
            # https://github.com/rubocop/rubocop-ast/blob/master/lib/rubocop/ast/node_pattern/method_definer.rb
            if source.start_with?("def_node_matcher")
              # https://github.com/Shopify/tapioca/blob/3341a9b/lib/tapioca/rbi_ext/model.rb#L89
              klass.create_method(
                method_name.to_s,
                parameters:  [
                  create_rest_param("node", type: "RuboCop::AST::Node"),
                  create_kw_rest_param("kwargs", type: "T.untyped"),
                  create_block_param("block", type: "T.untyped"),
                ],
                return_type: "T.untyped",
              )
            elsif source.start_with?("def_node_search")
              klass.create_method(
                method_name.to_s,
                parameters:  [
                  create_rest_param("node", type: "T.untyped"),
                  create_block_param("block", type: "T.untyped"),
                ],
                return_type: method_name.to_s.end_with?("?") ? "T::Boolean" : "T.untyped",
              )
            end
          end
        end
      end
    end
  end
end
