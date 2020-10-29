# typed: strict
# frozen_string_literal: true

module FormulaCellarChecks
  extend T::Sig
  sig { params(filename: Pathname).returns(T::Boolean) }
  def valid_library_extension?(filename)
    generic_valid_library_extension?(filename) || filename.basename.to_s.include?(".so.")
  end
end
