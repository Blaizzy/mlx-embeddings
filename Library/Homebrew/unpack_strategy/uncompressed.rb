# typed: false
# frozen_string_literal: true

module UnpackStrategy
  # Strategy for unpacking uncompressed files.
  class Uncompressed
    extend T::Sig

    include UnpackStrategy

    def extract_nestedly(prioritise_extension: false, **options)
      extract(**options)
    end

    private

    sig { override.params(unpack_dir: Pathname, basename: Pathname, verbose: T::Boolean).returns(T.untyped) }
    def extract_to_dir(unpack_dir, basename:, verbose:)
      FileUtils.cp path, unpack_dir/basename, preserve: true, verbose: verbose
    end
  end
end
