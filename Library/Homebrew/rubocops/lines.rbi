# typed: strict

module RuboCop
  module Cop
    module FormulaAudit
      class OnSystemConditionals < FormulaCop
        sig { params(node: T.any, on_method: Symbol, block: T.proc.params(parameters: T::Array[T.any]).void).void }
        def on_macos_version_method_call(node, on_method:, &block); end

        sig { params(node: T.any, block: T.proc.params(macos_symbol: Symbol).void).void }
        def on_system_method_call(node, &block); end

        sig { params(node: T.any, arch: Symbol, block: T.proc.params(node: T.any, else_node: T.any).void).void }
        def if_arch_node_search(node, arch:, &block); end

        sig { params(node: T.any, base_os: Symbol, block: T.proc.params(node: T.any, else_node: T.any).void).void }
        def if_base_os_node_search(node, base_os:, &block); end

        sig { params(node: T.any, os_name: Symbol, block: T.proc.params(node: T.any, operator: Symbol, else_node: T.any).void).void }
        def if_macos_version_node_search(node, os_name:, &block); end
      end
    end
  end
end
