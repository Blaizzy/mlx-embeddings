# typed: true
# frozen_string_literal: true

module Homebrew
  # Auditor for checking common violations in {Tap}s.
  #
  # @api private
  class TapAuditor
    extend T::Sig

    attr_reader :name, :path, :tap_audit_exceptions, :problems

    sig { params(tap: Tap, strict: T.nilable(T::Boolean)).void }
    def initialize(tap, strict:)
      @name                 = tap.name
      @path                 = tap.path
      @tap_audit_exceptions = tap.audit_exceptions
      @problems             = []
    end

    sig { void }
    def audit
      audit_json_files
      audit_tap_audit_exceptions
    end

    sig { void }
    def audit_json_files
      json_patterns = Tap::HOMEBREW_TAP_JSON_FILES.map { |pattern| @path/pattern }
      Pathname.glob(json_patterns).each do |file|
        JSON.parse file.read
      rescue JSON::ParserError
        problem "#{file.to_s.delete_prefix("#{@path}/")} contains invalid JSON"
      end
    end

    sig { void }
    def audit_tap_audit_exceptions
      @tap_audit_exceptions.each do |list_name, formula_names|
        unless [Hash, Array].include? formula_names.class
          problem <<~EOS
            audit_exceptions/#{list_name}.json should contain a JSON array
            of formula names or a JSON object mapping formula names to values
          EOS
          next
        end

        formula_names = formula_names.keys if formula_names.is_a? Hash

        invalid_formulae = []
        formula_names.each do |name|
          invalid_formulae << name if Formula[name].tap != @name
        rescue FormulaUnavailableError
          invalid_formulae << name
        end

        next if invalid_formulae.empty?

        problem <<~EOS
          audit_exceptions/#{list_name}.json references
          formulae that are not found in the #{@name} tap.
          Invalid formulae: #{invalid_formulae.join(", ")}
        EOS
      end
    end

    sig { params(message: String).void }
    def problem(message)
      @problems << ({ message: message, location: nil })
    end
  end
end
