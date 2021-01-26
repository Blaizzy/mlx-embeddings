# typed: false
# frozen_string_literal: true

require "cask/dsl"

module Cask
  class Cmd
    # Implementation of the `brew cask _stanza` command.
    #
    # @api private
    class InternalStanza < AbstractInternalCommand
      extend T::Sig

      # Syntax
      #
      #     brew cask _stanza <stanza_name> [ --quiet ] [ --table | --yaml ] [ <cask_token> ... ]
      #
      # If no tokens are given, then data for all casks is returned.
      #
      # The pseudo-stanza "artifacts" is available.
      #
      # On failure, a blank line is returned on the standard output.
      #

      ARTIFACTS =
        (DSL::ORDINARY_ARTIFACT_CLASSES.map(&:dsl_key) +
         DSL::ARTIFACT_BLOCK_CLASSES.map(&:dsl_key)).freeze

      sig { override.returns(T.nilable(T.any(Integer, Symbol))) }
      def self.min_named
        1
      end

      sig { returns(String) }
      def self.banner_args
        " <stanza_name> [<cask>]"
      end

      sig { returns(String) }
      def self.description
        <<~EOS
          Extract and render a specific stanza for the given <cask>.

          Examples:
            `brew cask _stanza appcast   --table`
            `brew cask _stanza app       --table           alfred google-chrome vagrant`
            `brew cask _stanza url       --table           alfred google-chrome vagrant`
            `brew cask _stanza version   --table           alfred google-chrome vagrant`
            `brew cask _stanza artifacts --table           alfred google-chrome vagrant`
            `brew cask _stanza artifacts --table --yaml    alfred google-chrome vagrant`
        EOS
      end

      def self.parser
        super do
          switch "--table",
                 description: "Print stanza in table format."
          switch "--quiet",
                 description: ""
          switch "--yaml",
                 description: ""
          switch "--inspect",
                 description: ""
        end
      end

      attr_accessor :format, :stanza
      private :format, :format=
      private :stanza, :stanza=

      def initialize(*)
        super

        named = args.named.dup
        @stanza = named.shift.to_sym
        args.freeze_named_args!(named)

        @format = :to_yaml if args.yaml?

        return if DSL::DSL_METHODS.include?(stanza)

        raise UsageError, <<~EOS
          Unknown/unsupported stanza '#{stanza}'.
          Check cask reference for supported stanzas.
        EOS
      end

      sig { void }
      def run
        if ARTIFACTS.include?(stanza)
          artifact_name = stanza
          @stanza = :artifacts
        end

        casks(alternative: -> { Cask.to_a }).each do |cask|
          print "#{cask}\t" if args.table?

          begin
            value = cask.send(stanza)
          rescue
            opoo "Failure calling '#{stanza}' on Cask '#{cask}'" unless args.quiet?
            puts ""
            next
          end

          if stanza == :artifacts
            value = Hash[value.map { |v| [v.class.dsl_key, v.to_s] }]
            value = value[artifact_name] if artifact_name
          end

          if value.nil? || (value.respond_to?(:empty?) && value.empty?)
            stanza_name = artifact_name || stanza
            raise CaskError, "no such stanza '#{stanza_name}' on Cask '#{cask}'"
          end

          if format
            puts value.send(format)
          elsif value.is_a?(Symbol)
            puts value.inspect
          else
            puts value.to_s
          end
        end
      end
    end
  end
end
