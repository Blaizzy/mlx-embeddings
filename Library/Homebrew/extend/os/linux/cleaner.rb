# typed: strict
# frozen_string_literal: true

class Cleaner
  private

  extend T::Sig
  sig { params(path: Pathname).returns(T.nilable(T::Boolean)) }
  def executable_path?(path)
    path.elf? || path.text_executable?
  end
end
