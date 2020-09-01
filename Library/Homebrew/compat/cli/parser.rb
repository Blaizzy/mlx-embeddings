# frozen_string_literal: true

module Homebrew
  module CLI
    class Parser
      module Compat
        module DeprecatedArgs
          # No need to define it as its the default/superclass implementation.
          # rubocop:disable Style/MissingRespondToMissing
          def method_missing(method, *)
            if ![:debug?, :quiet?, :verbose?, :value].include?(method) && !@printed_args_warning
              odeprecated "Homebrew.args", "`args = <command>_args.parse` and pass `args` along the call chain"
              @printed_args_warning = true
            end

            super
          end
          # rubocop:enable Style/MissingRespondToMissing
        end

        def parse(*)
          args = super
          Homebrew.args = args.dup.extend(DeprecatedArgs)
          args
        end
      end

      prepend Compat
    end
  end
end
