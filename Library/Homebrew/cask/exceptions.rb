# typed: true
# frozen_string_literal: true

module Cask
  # General cask error.
  #
  # @api private
  class CaskError < RuntimeError; end

  # Cask error containing multiple other errors.
  #
  # @api private
  class MultipleCaskErrors < CaskError
    extend T::Sig

    def initialize(errors)
      super()

      @errors = errors
    end

    sig { returns(String) }
    def to_s
      <<~EOS
        Problems with multiple casks:
        #{@errors.map(&:to_s).join("\n")}
      EOS
    end
  end

  # Abstract cask error containing a cask token.
  #
  # @api private
  class AbstractCaskErrorWithToken < CaskError
    extend T::Sig

    sig { returns(String) }
    attr_reader :token

    sig { returns(String) }
    attr_reader :reason

    def initialize(token, reason = nil)
      super()

      @token = token.to_s
      @reason = reason.to_s
    end
  end

  # Error when a cask is not installed.
  #
  # @api private
  class CaskNotInstalledError < AbstractCaskErrorWithToken
    extend T::Sig

    sig { returns(String) }
    def to_s
      "Cask '#{token}' is not installed."
    end
  end

  # Error when a cask conflicts with another cask.
  #
  # @api private
  class CaskConflictError < AbstractCaskErrorWithToken
    extend T::Sig

    attr_reader :conflicting_cask

    def initialize(token, conflicting_cask)
      super(token)
      @conflicting_cask = conflicting_cask
    end

    sig { returns(String) }
    def to_s
      "Cask '#{token}' conflicts with '#{conflicting_cask}'."
    end
  end

  # Error when a cask is not available.
  #
  # @api private
  class CaskUnavailableError < AbstractCaskErrorWithToken
    extend T::Sig

    sig { returns(String) }
    def to_s
      "Cask '#{token}' is unavailable#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask is unreadable.
  #
  # @api private
  class CaskUnreadableError < CaskUnavailableError
    extend T::Sig

    sig { returns(String) }
    def to_s
      "Cask '#{token}' is unreadable#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask already exists.
  #
  # @api private
  class CaskAlreadyCreatedError < AbstractCaskErrorWithToken
    extend T::Sig

    sig { returns(String) }
    def to_s
      %Q(Cask '#{token}' already exists. Run #{Formatter.identifier("brew edit --cask #{token}")} to edit it.)
    end
  end

  # Error when a cask is already installed.
  #
  # @api private
  class CaskAlreadyInstalledError < AbstractCaskErrorWithToken
    extend T::Sig

    sig { returns(String) }
    def to_s
      <<~EOS
        Cask '#{token}' is already installed.

        To re-install #{token}, run:
          #{Formatter.identifier("brew reinstall #{token}")}
      EOS
    end
  end

  # Error when a cask depends on X11.
  #
  # @api private
  class CaskX11DependencyError < AbstractCaskErrorWithToken
    extend T::Sig

    sig { returns(String) }
    def to_s
      <<~EOS
        Cask '#{token}' requires XQuartz/X11, which can be installed using Homebrew Cask by running:
          #{Formatter.identifier("brew install --cask xquartz")}

        or manually, by downloading the package from:
          #{Formatter.url("https://www.xquartz.org/")}
      EOS
    end
  end

  # Error when there is a cyclic cask dependency.
  #
  # @api private
  class CaskCyclicDependencyError < AbstractCaskErrorWithToken
    extend T::Sig

    sig { returns(String) }
    def to_s
      "Cask '#{token}' includes cyclic dependencies on other Casks#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask depends on itself.
  #
  # @api private
  class CaskSelfReferencingDependencyError < CaskCyclicDependencyError
    extend T::Sig

    sig { returns(String) }
    def to_s
      "Cask '#{token}' depends on itself."
    end
  end

  # Error when no cask is specified.
  #
  # @api private
  class CaskUnspecifiedError < CaskError
    extend T::Sig

    sig { returns(String) }
    def to_s
      "This command requires a Cask token."
    end
  end

  # Error when a cask is invalid.
  #
  # @api private
  class CaskInvalidError < AbstractCaskErrorWithToken
    extend T::Sig

    sig { returns(String) }
    def to_s
      "Cask '#{token}' definition is invalid#{reason.empty? ? "." : ": #{reason}"}"
    end
  end

  # Error when a cask token does not match the file name.
  #
  # @api private
  class CaskTokenMismatchError < CaskInvalidError
    def initialize(token, header_token)
      super(token, "Token '#{header_token}' in header line does not match the file name.")
    end
  end

  # Error during quarantining of a file.
  #
  # @api private
  class CaskQuarantineError < CaskError
    extend T::Sig

    attr_reader :path, :reason

    def initialize(path, reason)
      super()

      @path = path
      @reason = reason
    end

    sig { returns(String) }
    def to_s
      s = +"Failed to quarantine #{path}."

      unless reason.empty?
        s << " Here's the reason:\n"
        s << Formatter.error(reason)
        s << "\n" unless reason.end_with?("\n")
      end

      s.freeze
    end
  end

  # Error while propagating quarantine information to subdirectories.
  #
  # @api private
  class CaskQuarantinePropagationError < CaskQuarantineError
    extend T::Sig

    sig { returns(String) }
    def to_s
      s = +"Failed to quarantine one or more files within #{path}."

      unless reason.empty?
        s << " Here's the reason:\n"
        s << Formatter.error(reason)
        s << "\n" unless reason.end_with?("\n")
      end

      s.freeze
    end
  end

  # Error while removing quarantine information.
  #
  # @api private
  class CaskQuarantineReleaseError < CaskQuarantineError
    extend T::Sig

    sig { returns(String) }
    def to_s
      s = +"Failed to release #{path} from quarantine."

      unless reason.empty?
        s << " Here's the reason:\n"
        s << Formatter.error(reason)
        s << "\n" unless reason.end_with?("\n")
      end

      s.freeze
    end
  end
end
