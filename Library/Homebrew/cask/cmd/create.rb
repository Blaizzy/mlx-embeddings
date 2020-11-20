# typed: true
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask create` command.
    #
    # @api private
    class Create < AbstractCommand
      extend T::Sig

      sig { override.returns(T.nilable(T.any(Integer, Symbol))) }
      def self.min_named
        :cask
      end

      sig { override.returns(T.nilable(Integer)) }
      def self.max_named
        1
      end

      sig { returns(String) }
      def self.description
        "Creates the given <cask> and opens it in an editor."
      end

      def initialize(*)
        super
      rescue Homebrew::CLI::MaxNamedArgumentsError
        raise UsageError, "Only one cask can be created at a time."
      end

      sig { void }
      def run
        cask_token = args.named.first
        cask_path = CaskLoader.path(cask_token)
        raise CaskAlreadyCreatedError, cask_token if cask_path.exist?

        odebug "Creating Cask #{cask_token}"
        File.open(cask_path, "w") do |f|
          f.write self.class.template(cask_token)
        end

        exec_editor cask_path
      end

      def self.template(cask_token)
        <<~RUBY
          cask "#{cask_token}" do
            version ""
            sha256 ""

            url "https://"
            name ""
            desc ""
            homepage ""

            app ""
          end
        RUBY
      end
    end
  end
end
