# typed: strict

module Homebrew
  module CLI
    class Parser
      module Compat
        include Kernel

        module DeprecatedArgs
          include Kernel
        end
      end
    end
  end
end
