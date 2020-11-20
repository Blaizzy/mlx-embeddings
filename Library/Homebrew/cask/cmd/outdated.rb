# typed: false
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask outdated` command.
    #
    # @api private
    class Outdated < AbstractCommand
      extend T::Sig

      sig { returns(String) }
      def self.description
        "List the outdated installed casks."
      end

      def self.parser
        super do
          switch "--greedy",
                 description: "Also include casks which specify `auto_updates true` or `version :latest`."
          switch "--json",
                 description: "Print a JSON representation of outdated casks."
        end
      end

      sig { void }
      def run
        outdated_casks = casks(alternative: -> { Caskroom.casks(config: Config.from_args(args)) }).select do |cask|
          odebug "Checking update info of Cask #{cask}"
          cask.outdated?(greedy: args.greedy?)
        end

        verbose = ($stdout.tty? || args.verbose?) && !args.quiet?
        output = outdated_casks.map { |cask| cask.outdated_info(args.greedy?, verbose, args.json?) }

        puts args.json? ? JSON.generate(output) : output
      end
    end
  end
end
